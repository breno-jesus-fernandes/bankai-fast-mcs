use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn criterion_harness_smoke(c: &mut Criterion) {
    c.bench_function("criterion_harness_smoke", |b| {
        b.iter(|| black_box(1_usize + 1))
    });
}

criterion_group!(benches, criterion_harness_smoke);
criterion_main!(benches);
