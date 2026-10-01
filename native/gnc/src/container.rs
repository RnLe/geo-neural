//! The `.gnc` container (`geoneural.codecs.package`). Little-endian:
//!
//! ```text
//! magic "GNC1", version u16, flags u16
//! rows u32, cols u32, lattice_m f64, E u32
//! west f64, north f64, spacing_m f64, horizontal EPSG u32, vertical EPSG u32, node-centred u8
//! coder u8, foreign codec id (8 bytes ASCII), model sha256 (32 bytes), rANS table id (8 bytes)
//! component count u8, then per component: kind u8, length u32, crc32 u32
//! payloads in directory order
//! ```

use crate::tables::TABLE_ID;
use crate::{fail, Result};

pub const MAGIC: &[u8; 4] = b"GNC1";
pub const VERSION: u16 = 1;
pub const FLAG_MODEL_EMBEDDED: u16 = 1;
pub const HEADER_BYTES: usize = 111;
pub const ENTRY_BYTES: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coder {
    CubicOrder0,
    CubicCtx,
    Learned,
    Foreign,
}

impl Coder {
    fn from_code(code: u8) -> Option<Coder> {
        match code {
            0 => Some(Coder::CubicOrder0),
            1 => Some(Coder::CubicCtx),
            2 => Some(Coder::Learned),
            16 => Some(Coder::Foreign),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Coder::CubicOrder0 => "cubic-order0",
            Coder::CubicCtx => "cubic-ctx",
            Coder::Learned => "learned",
            Coder::Foreign => "foreign",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Coarse,
    Params,
    Stream,
    Raw,
    Model,
    Context,
    Mask,
    Rule,
    Foreign,
}

impl Kind {
    fn from_code(code: u8) -> Option<Kind> {
        match code {
            1 => Some(Kind::Coarse),
            2 => Some(Kind::Params),
            3 => Some(Kind::Stream),
            4 => Some(Kind::Raw),
            5 => Some(Kind::Model),
            6 => Some(Kind::Context),
            7 => Some(Kind::Mask),
            8 => Some(Kind::Rule),
            20 => Some(Kind::Foreign),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Coarse => "coarse",
            Kind::Params => "params",
            Kind::Stream => "stream",
            Kind::Raw => "raw",
            Kind::Model => "model",
            Kind::Context => "context",
            Kind::Mask => "mask",
            Kind::Rule => "rule",
            Kind::Foreign => "foreign",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Component<'a> {
    pub kind: Kind,
    pub data: &'a [u8],
}

/// A product whose header and components passed every check; payloads borrow from the input.
#[derive(Debug, Clone)]
pub struct Product<'a> {
    pub version: u16,
    pub flags: u16,
    pub rows: u32,
    pub cols: u32,
    pub lattice_m: f64,
    pub e: u32,
    pub west: f64,
    pub north: f64,
    pub spacing_m: f64,
    pub epsg_h: u32,
    pub epsg_v: u32,
    pub coder: Coder,
    pub foreign: String,
    /// None when the header holds zeros (no learned model).
    pub model_sha256: Option<[u8; 32]>,
    pub model_embedded: bool,
    pub components: Vec<Component<'a>>,
}

impl<'a> Product<'a> {
    /// Payload of a component kind; with repeated kinds the last one wins, as in the Python reader.
    pub fn component(&self, kind: Kind) -> Option<&'a [u8]> {
        self.components
            .iter()
            .rev()
            .find(|c| c.kind == kind)
            .map(|c| c.data)
    }

    /// Bytes per part: the container (header and directory), then every component in file order.
    pub fn breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut out = vec![(
            "container",
            HEADER_BYTES + ENTRY_BYTES * self.components.len(),
        )];
        out.extend(
            self.components
                .iter()
                .map(|c| (c.kind.name(), c.data.len())),
        );
        out
    }

    pub fn total_bytes(&self) -> usize {
        self.breakdown().iter().map(|(_, n)| n).sum()
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.off..self.off + N]);
        self.off += N;
        out
    }

    fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }

    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take())
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }

    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take())
    }
}

/// Parses and checks a product, in the order of `package.read`.
pub fn read(blob: &[u8]) -> Result<Product<'_>> {
    if blob.len() < HEADER_BYTES {
        return fail("file shorter than the header");
    }
    let mut c = Cursor {
        bytes: blob,
        off: 0,
    };
    let magic: [u8; 4] = c.take();
    let version = c.u16();
    let flags = c.u16();
    let rows = c.u32();
    let cols = c.u32();
    let lattice_m = c.f64();
    let e = c.u32();
    let west = c.f64();
    let north = c.f64();
    let spacing_m = c.f64();
    let epsg_h = c.u32();
    let epsg_v = c.u32();
    let node = c.u8();
    let coder_code = c.u8();
    let foreign: [u8; 8] = c.take();
    let model: [u8; 32] = c.take();
    let table: [u8; 8] = c.take();
    let count = c.u8() as usize;
    debug_assert_eq!(c.off, HEADER_BYTES);

    if &magic != MAGIC {
        return fail("not a GNC product");
    }
    if version != VERSION {
        return fail(format!("product version {version} is not supported"));
    }
    let Some(coder) = Coder::from_code(coder_code) else {
        return fail(format!("unknown coder {coder_code}"));
    };
    if node != 1 {
        return fail("only node-centred lattices are defined");
    }
    if coder != Coder::Foreign && table != TABLE_ID {
        return fail("product was written with other rANS tables");
    }
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if c.off + ENTRY_BYTES > blob.len() {
            return fail("truncated component directory");
        }
        let code = c.u8();
        let length = c.u32() as usize;
        let crc = c.u32();
        let Some(kind) = Kind::from_code(code) else {
            return fail(format!("unknown component kind {code}"));
        };
        entries.push((kind, length, crc));
    }
    let mut off = c.off;
    let mut components = Vec::with_capacity(count);
    for (kind, length, crc) in entries {
        let end = off.saturating_add(length);
        if end > blob.len() {
            return fail(format!("component {} is truncated", kind.name()));
        }
        let data = &blob[off..end];
        if crc32fast::hash(data) != crc {
            return fail(format!("component {} fails its checksum", kind.name()));
        }
        components.push(Component { kind, data });
        off = end;
    }
    if off != blob.len() {
        return fail("bytes after the last component");
    }
    let end = foreign.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    let Ok(foreign) = String::from_utf8(foreign[..end].to_vec()) else {
        return fail("the foreign codec id is not UTF-8 text");
    };
    Ok(Product {
        version,
        flags,
        rows,
        cols,
        lattice_m,
        e,
        west,
        north,
        spacing_m,
        epsg_h,
        epsg_v,
        coder,
        foreign,
        model_sha256: if model == [0u8; 32] {
            None
        } else {
            Some(model)
        },
        model_embedded: flags & FLAG_MODEL_EMBEDDED != 0,
        components,
    })
}
