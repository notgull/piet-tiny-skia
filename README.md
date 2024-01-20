# piet-tiny-skia

A `piet` frontend for the `tiny-skia` framework.

`tiny-skia` is a very high-quality implementation of software rendering, based
on the algorithms used by Skia. It is the fastest software renderer in the Rust
community and it produces high-quality results. However, the feature set of the
crate is intentionally limited so that what is there is fast and correct.

This crate, `piet-tiny-skia`, provides a `piet`-based frontend for `tiny-skia`
that may be more familiar to users of popular graphics APIs. It may be easier to
use than the `tiny-skia` API while also maintaining the flexibility. It also
provides text rendering, provided by the
`cosmic-text` crate.

To start, create a `tiny_skia::PixmapMut` and a `Cache`. Then, use the `Cache`
to create a `RenderContext` to render into the pixmap. Finally the pixmap can be
saved to a file or rendered elsewhere.

## Source Code

The canonical code for this repository is kept on [Codeberg]. For
convenience, a mirror is kept on [GitHub].

[Codeberg]: https://codeberg.org/notgull/piet-cosmic-text
[GitHub]: https://github.com/notgull/piet-cosmic-text

## License

`piet-tiny-skia` is free software: you can redistribute it and/or modify it under
the terms of either:

* GNU Lesser General Public License as published by the Free Software Foundation,
  either version 3 of the License, or (at your option) any later version.
* Mozilla Public License as published by the Mozilla Foundation, version 2.

`piet-tiny-skia` is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
Mozilla Public License for more details.

You should have received a copy of the GNU Lesser General Public License and the
Mozilla Public License along with `piet-tiny-skia`. If not, see
<https://www.gnu.org/licenses/> or <https://www.mozilla.org/en-US/MPL/2.0/>.
