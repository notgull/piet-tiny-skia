# piet-tiny-skia Changelog

## v0.2.3

- Move source code to codeberg.org

## v0.2.2

- Move source code to src.notgull.net

## v0.2.1

- Add `target()`, `target_mut()`, and `into_target()` to `RenderContext` to
  allow for access of the inner target.

## v0.2.0

- **Breaking:** `RenderContext` now wraps around a `T: AsPixmapMut` instead of
  a direct `PixmapMut`.

## v0.1.0

Initial release.
