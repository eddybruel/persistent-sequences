use std::ops::{Bound, Range, RangeBounds};

pub fn check_bounds<R: RangeBounds<usize>>(range: R, len: usize) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(start) => *start,
        Bound::Excluded(start) => start.checked_add(1).unwrap(),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(end) => end.checked_add(1).unwrap(),
        Bound::Excluded(end) => *end,
        Bound::Unbounded => len,
    };
    assert!(start <= end);
    assert!(end <= len);
    Range { start, end }
}
