// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-tiny-skia`.
//
// `piet-tiny-skia` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
// * The Patron License (https://github.com/notgull/piet-tiny-skia/blob/main/LICENSE-PATRON.md)
//   for sponsors and contributors, who can ignore the copyleft provisions of the above licenses
//   for this project.
//
// `piet-tiny-skia` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-tiny-skia`. If not, see <https://www.gnu.org/licenses/>.

use piet::samples;
use tiny_skia as ts;

fn main() {
    samples::samples_main(
        |number, scale, path| {
            let picture = samples::get(number)?;
            let size = picture.size();
            let scaled_width = (size.width * scale) as u32;
            let scaled_height = (size.height * scale) as u32;

            let mut pixmap = ts::Pixmap::new(scaled_width, scaled_height).unwrap();
            let mut cache = piet_tiny_skia::Cache::new();

            let mut rc = cache.render_context(pixmap.as_mut());
            rc.set_bitmap_scale(scale);
            piet::RenderContext::text(&mut rc).set_dpi(72.0);

            // Render the picture.
            picture.draw(&mut rc)?;

            // Save to the path.
            pixmap.save_png(path)?;

            Ok(())
        },
        "piet-tiny-skia",
        None,
    )
}
