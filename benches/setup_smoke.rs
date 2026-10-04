use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn criterion_harness_smoke(c: &mut Criterion) {
    let values: Vec<u64> = (0..1024).collect();
    c.bench_function("criterion_harness_reduction", |b| {
        b.iter(|| black_box(black_box(&values).iter().copied().sum::<u64>()))
    });
}

criterion_group!(benches, criterion_harness_smoke);
criterion_main!(benches);
