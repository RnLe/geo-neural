//! Bit-exact parity with the Python decoder on the fixtures written by
//! `geoneural.codecs.fixtures.write_fixtures`, and every container check on damaged copies.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn table(name: &str) -> Vec<Vec<String>> {
    let text = String::from_utf8(load(name)).unwrap();
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

fn model(name: &str) -> Option<Vec<u8>> {
    (name != "-").then(|| load(name))
}

fn sha_hex(bytes: impl IntoIterator<Item = [u8; 8]>) -> String {
    let mut h = Sha256::new();
    for b in bytes {
        h.update(b);
    }
    gnc::hex(&h.finalize())
}

fn unzstd(blob: &[u8]) -> Vec<u8> {
    let mut src = blob;
    let mut out = Vec::new();
    ruzstd::decoding::StreamingDecoder::new(&mut src)
        .unwrap()
        .read_to_end(&mut out)
        .unwrap();
    out
}

#[test]
fn every_fixture_decodes_bit_exactly() {
    let cases = table("cases.tsv");
    assert!(
        cases.len() >= 20,
        "fixtures missing; run geoneural.codecs.fixtures.write_fixtures"
    );
    for row in &cases {
        let [name, product, model_file, rows, cols, coder, e, lattice, lattice_sha, heights_sha] =
            &row[..]
        else {
            panic!("bad row {row:?}");
        };
        let blob = load(product);
        let m = model(model_file);
        let t = Instant::now();
        let field = gnc::decode(&blob, m.as_deref()).unwrap_or_else(|err| panic!("{name}: {err}"));
        let elapsed = t.elapsed();
        assert_eq!(field.rows.to_string(), *rows, "{name}");
        assert_eq!(field.cols.to_string(), *cols, "{name}");
        assert_eq!(field.coder.name(), coder, "{name}");
        assert_eq!(field.bound_units.to_string(), *e, "{name}");
        let got = field.lattice();
        if lattice != "-" {
            let want: Vec<i64> = unzstd(&load(lattice))
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(want.len(), got.len(), "{name}");
            if let Some(i) = (0..want.len()).find(|&i| want[i] != got[i]) {
                let side = field.cols;
                panic!(
                    "{name}: first difference at row {} col {}: Python {} Rust {}",
                    i / side,
                    i % side,
                    want[i],
                    got[i]
                );
            }
        }
        assert_eq!(
            &gnc::hex(&field.lattice_sha256()),
            lattice_sha,
            "{name}: lattice checksum"
        );
        assert_eq!(
            &sha_hex(got.iter().map(|v| v.to_le_bytes())),
            lattice_sha,
            "{name}"
        );
        assert_eq!(
            &sha_hex(field.heights_f64().iter().map(|v| v.to_le_bytes())),
            heights_sha,
            "{name}: metres"
        );
        if field.rows >= 1025 {
            eprintln!(
                "{name}: {} x {} decoded in {:.0} ms (unqualified)",
                field.rows,
                field.cols,
                elapsed.as_secs_f64() * 1e3
            );
        }
    }
}

#[test]
fn unsupported_products_are_refused() {
    for row in table("errors.tsv") {
        let [name, product, model_file, needle] = &row[..] else {
            panic!("bad row {row:?}")
        };
        let m = model(model_file);
        let err = gnc::decode(&load(product), m.as_deref()).expect_err(name);
        assert!(err.message().contains(needle.as_str()), "{name}: {err}");
    }
}

fn expect(blob: &[u8], model: Option<&[u8]>, needle: &str) {
    match gnc::decode(blob, model) {
        Ok(_) => panic!("decoded a product that should fail with '{needle}'"),
        Err(e) => assert!(
            e.message().contains(needle),
            "expected '{needle}', got '{e}'"
        ),
    }
}

fn case(prefix: &str) -> Vec<String> {
    table("cases.tsv")
        .into_iter()
        .find(|r| r[0] == prefix)
        .unwrap_or_else(|| panic!("no fixture {prefix}"))
}

#[test]
fn damaged_products_fail_their_checks() {
    let blob = load(&case("ridges-65-ctx")[1]);
    expect(&blob[..50], None, "shorter than the header");
    let mut b = blob.clone();
    b[0] = b'X';
    expect(&b, None, "not a GNC product");
    let mut b = blob.clone();
    b[4] = 2;
    expect(&b, None, "version 2");
    let mut b = blob.clone();
    b[61] = 9;
    expect(&b, None, "unknown coder 9");
    let mut b = blob.clone();
    b[60] = 0;
    expect(&b, None, "node-centred");
    let mut b = blob.clone();
    b[102] ^= 1;
    expect(&b, None, "other rANS tables");
    let mut b = blob.clone();
    b[111] = 99;
    expect(&b, None, "unknown component kind 99");
    let mut b = blob.clone();
    let last = b.len() - 1;
    b[last] ^= 0x40;
    expect(&b, None, "fails its checksum");
    expect(&blob[..blob.len() - 1], None, "is truncated");
    let mut b = blob.clone();
    b.push(0);
    expect(&b, None, "bytes after the last component");
    expect(&blob[..111 + 9 + 4], None, "truncated component directory");
    let mut b = blob.clone();
    b[12..16].copy_from_slice(&66u32.to_le_bytes());
    expect(&b, None, "square products only");
    gnc::decode(&blob, None).unwrap();
}

#[test]
fn models_are_checked_by_hash() {
    let corpus = case("ridges-65-learned-corpus");
    let blob = load(&corpus[1]);
    let v2 = load(&corpus[2]);
    let v1 = load(&case("ridges-65-learned-v1")[2]);
    assert_ne!(v1, v2);
    expect(&blob, None, "needs the shared model");
    let a = gnc::decode(&blob, Some(&v2)).unwrap();
    let b = gnc::decode(&blob, Some(&v1)).unwrap();
    assert_eq!(a.lattice(), b.lattice());
    let other = load(&case("essen-257-trained")[2]);
    expect(
        &blob,
        Some(&other),
        "not the one this product was encoded with",
    );
    let mut damaged = v2.clone();
    let n = damaged.len();
    damaged[n - 1] ^= 1;
    expect(
        &blob,
        Some(&damaged),
        "not the one this product was encoded with",
    );
    expect(&blob, Some(&v2[..n - 1]), "truncated");
    let mut extra = v2.clone();
    extra.extend_from_slice(&[0, 0]);
    expect(&blob, Some(&extra), "trailing or missing bytes");
    let mut table = v2.clone();
    table[18] ^= 1;
    expect(&blob, Some(&table), "other rANS tables");
    expect(&blob, Some(b"GNM1\x03\x00"), "version 3 not supported");
    // A standalone product carries its model; a given one is ignored.
    let standalone = load(&case("ridges-65-learned")[1]);
    assert_eq!(
        gnc::decode(&standalone, Some(&other)).unwrap().lattice(),
        a.lattice()
    );
}

#[test]
fn describe_reports_the_breakdown() {
    let corpus = case("ridges-65-learned-corpus");
    let blob = load(&corpus[1]);
    let text = gnc::describe(&blob).unwrap();
    let prod = gnc::container::read(&blob).unwrap();
    assert_eq!(prod.total_bytes(), blob.len());
    assert!(text.starts_with("{\"coder\":\"learned\""), "{text}");
    assert!(text.contains("[\"container\",147]"), "{text}");
    assert!(text.contains("\"modelEmbedded\":false"), "{text}");
}

/// Damaged payloads with repaired checksums reach the decoder proper; it must fail cleanly, never panic (a
/// panic aborts the WebAssembly module).
#[test]
fn damaged_payloads_never_panic() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for name in [
        "ridges-65-learned",
        "ridges-65-ctx",
        "tiny-9-ctx",
        "tiny-3-order0",
    ] {
        let blob = load(&case(name)[1]);
        let count = blob[110] as usize;
        let mut spans = Vec::new();
        let mut off = 111 + 9 * count;
        for k in 0..count {
            let e = 111 + 9 * k;
            let len = u32::from_le_bytes(blob[e + 1..e + 5].try_into().unwrap()) as usize;
            spans.push((e, off, off + len));
            off += len;
        }
        let mut failures = 0;
        for _ in 0..1500 {
            let mut b = blob.clone();
            for _ in 0..1 + next() % 3 {
                let at = (next() as usize) % b.len();
                b[at] ^= 1 + (next() % 255) as u8;
            }
            for &(entry, start, end) in &spans {
                let crc = crc32fast::hash(&b[start..end]);
                b[entry + 5..entry + 9].copy_from_slice(&crc.to_le_bytes());
            }
            if gnc::decode(&b, None).is_err() {
                failures += 1;
            }
        }
        assert!(failures > 0, "{name}: no damage was detected");
    }
}
