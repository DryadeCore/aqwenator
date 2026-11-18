/// Free state slots and KV pages. A page is owned by every running sequence, parked entry and
/// snapshot that lists it, and returns to the pool when its last owner releases it.
pub(super) struct Pool {
    pub(super) slots: Vec<u32>,
    pub(super) pages: Vec<u32>,
    refs: Vec<u32>,
}

impl Pool {
    pub(super) fn new(slots: usize, pages: usize) -> Self {
        Self { slots: (0..slots as u32).rev().collect(), pages: (0..pages as u32).rev().collect(), refs: vec![0; pages] }
    }

    pub(super) fn take_page(&mut self) -> Option<u32> {
        let p = self.pages.pop()?;
        self.refs[p as usize] = 1;
        Some(p)
    }

    pub(super) fn take_pages(&mut self, n: usize) -> Option<Vec<u32>> {
        if self.pages.len() < n {
            return None;
        }
        let out = self.pages.split_off(self.pages.len() - n);
        for &p in &out {
            self.refs[p as usize] = 1;
        }
        Some(out)
    }

    /// Add an owner to each page.
    pub(super) fn share(&mut self, pages: &[u32]) {
        for &p in pages {
            self.refs[p as usize] += 1;
        }
    }

    /// Drop an owner of each page.
    pub(super) fn release(&mut self, pages: impl IntoIterator<Item = u32>) {
        for p in pages {
            let r = &mut self.refs[p as usize];
            *r -= 1;
            if *r == 0 {
                self.pages.push(p);
            }
        }
    }

    pub(super) fn refs(&self, page: u32) -> u32 {
        self.refs[page as usize]
    }
}
