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

//! A [`piet`] frontend for the [`tiny-skia`] framework.

#![forbid(unsafe_code)]

pub use piet;
pub use tiny_skia;

use cosmic_text::{Command as ZenoCommand, SwashCache};
use piet::kurbo::{self, Affine, PathEl, Shape};
use piet::Error as Pierror;

use std::mem;
use std::slice;

use tiny_skia as ts;
use tinyvec::TinyVec;
use ts::{Mask, PathBuilder, PixmapMut, Shader};

/// The cache for [`tiny-skia`] resources.
pub struct Cache {
    /// Cached path builder.
    path_builder: Option<PathBuilder>,

    /// The text renderer object.
    text: Text,

    /// Drawn glyphs.
    glyph_cache: Option<SwashCache>,

    /// Allocation to hold dashes in.
    dash_buffer: Vec<f32>,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            path_builder: Some(PathBuilder::new()),
            text: Text(piet_cosmic_text::Text::new()),
            glyph_cache: Some(SwashCache::new()),
            dash_buffer: vec![],
        }
    }
}

/// The whole point.
pub struct RenderContext<'cache, 'target> {
    /// The mutable reference to the cache.
    cache: &'cache mut Cache,

    /// The mutable reference to the target.
    target: PixmapMut<'target>,

    /// The last error that occurred.
    last_error: Result<(), Pierror>,

    /// The stack of render states.
    states: TinyVec<[RenderState; 1]>,

    /// Tolerance for curves.
    tolerance: f64,

    /// Scale to apply for bitmaps.
    bitmap_scale: f64,

    /// Flag to ignore the current state.
    ignore_state: bool,
}

/// Rendering state frame.
struct RenderState {
    /// The current transform.
    transform: Affine,

    /// The current clip.
    clip: Option<Mask>,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            transform: Affine::IDENTITY,
            clip: None,
        }
    }
}

/// The text renderer for [`tiny-skia`].
#[derive(Debug, Clone)]
pub struct Text(piet_cosmic_text::Text);

/// The text layout builder for [`tiny-skia`].
#[derive(Debug)]
pub struct TextLayoutBuilder(piet_cosmic_text::TextLayoutBuilder);

/// The text layout for [`tiny-skia`].
#[derive(Debug, Clone)]
pub struct TextLayout(piet_cosmic_text::TextLayout);

/// The brush for [`tiny-skia`].
#[derive(Clone)]
pub struct Brush(BrushInner);

#[derive(Clone)]
enum BrushInner {
    /// Solid color.
    Solid(piet::Color),

    /// Fixed linear gradient brush.
    LinearGradient(piet::FixedLinearGradient),

    /// Fixed radial gradient brush.
    RadialGradient(piet::FixedRadialGradient),
}

/// The image used [`tiny-skia`].
#[derive(Clone)]
pub struct Image(tiny_skia::Pixmap);

impl Cache {
    /// Creates a new cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets a render context for the provided target.
    pub fn render_context<'cache, 'target>(
        &'cache mut self,
        target: PixmapMut<'target>,
    ) -> RenderContext<'cache, 'target> {
        RenderContext {
            cache: self,
            target,
            last_error: Ok(()),
            states: tinyvec::tiny_vec![RenderState::default(); 1],
            tolerance: 0.1,
            bitmap_scale: 1.0,
            ignore_state: false,
        }
    }
}

impl RenderContext<'_, '_> {
    /// Get the bitmap scale.
    pub fn bitmap_scale(&self) -> f64 {
        self.bitmap_scale
    }

    /// Set the bitmap scale.
    pub fn set_bitmap_scale(&mut self, scale: f64) {
        self.bitmap_scale = scale;
    }

    fn fill_impl(
        &mut self,
        shape: impl Shape,
        shader: tiny_skia::Shader<'_>,
        rule: tiny_skia::FillRule,
    ) {
        // Get the last state out.
        let Self {
            cache,
            target,
            states,
            tolerance,
            ..
        } = self;

        // Get out the path builder.
        let mut builder = cache.path_builder.take().unwrap_or_default();

        // Convert the shape to a path.
        cvt_shape_to_skia_path(&mut builder, shape, *tolerance);
        let path = match builder.finish() {
            Some(path) => path,
            None => return,
        };

        // Draw the shape.
        let paint = tiny_skia::Paint {
            shader,
            ..Default::default()
        };
        let state = states.last().unwrap();

        let (transform, mask) = if self.ignore_state {
            (
                ts::Transform::from_scale(self.bitmap_scale as f32, self.bitmap_scale as f32),
                None,
            )
        } else {
            let real_transform = Affine::scale(self.bitmap_scale) * state.transform;
            (cvt_affine(real_transform), state.clip.as_ref())
        };

        target.fill_path(&path, &paint, rule, transform, mask);

        // Keep the allocation around.
        self.cache.path_builder = Some(path.clear());
    }

    fn stroke_impl(
        &mut self,
        shape: impl Shape,
        shader: tiny_skia::Shader<'_>,
        stroke: &ts::Stroke,
    ) {
        // Get the last state out.
        let Self {
            cache,
            target,
            states,
            tolerance,
            ..
        } = self;

        // Get out the path builder.
        let mut builder = cache.path_builder.take().unwrap_or_default();

        // Convert the shape to a path.
        cvt_shape_to_skia_path(&mut builder, shape, *tolerance);
        let path = match builder.finish() {
            Some(path) => path,
            None => return,
        };

        // Draw the shape.
        let paint = tiny_skia::Paint {
            shader,
            ..Default::default()
        };
        let state = states.last().unwrap();

        let (transform, mask) = if self.ignore_state {
            (
                ts::Transform::from_scale(self.bitmap_scale as f32, self.bitmap_scale as f32),
                None,
            )
        } else {
            let real_transform = Affine::scale(self.bitmap_scale) * state.transform;
            (cvt_affine(real_transform), state.clip.as_ref())
        };

        target.stroke_path(&path, &paint, stroke, transform, mask);

        // Keep the allocation around.
        self.cache.path_builder = Some(path.clear());
    }

    #[allow(clippy::if_same_then_else)]
    fn draw_glyph(&mut self, pos: kurbo::Point, glyph: &cosmic_text::LayoutGlyph, run_y: f32) {
        // Take the glyph cache or make a new one.
        let mut glyph_cache = self
            .cache
            .glyph_cache
            .take()
            .unwrap_or_else(SwashCache::new);

        let physical = glyph.physical((0., 0.), 1.0);
        self.cache.text.clone().0.with_font_system_mut(|system| {
            // Try to get the font outline, which we can draw directly with tiny-skia.
            if let Some(outline) = glyph_cache.get_outline_commands(system, physical.cache_key) {
                let offset = kurbo::TranslateScale::new(
                    kurbo::Vec2::new(
                        pos.x + physical.x as f64 + physical.cache_key.x_bin.as_float() as f64,
                        pos.y
                            + run_y as f64
                            + physical.y as f64
                            + physical.cache_key.y_bin.as_float() as f64,
                    ),
                    1.0,
                );
                let color = glyph.color_opt.map_or(
                    {
                        let (r, g, b, a) = piet::util::DEFAULT_TEXT_COLOR.as_rgba();
                        ts::Color::from_rgba(r as f32, g as f32, b as f32, a as f32)
                            .expect("default text color should be valid")
                    },
                    |c| {
                        let [r, g, b, a] = [c.r(), c.g(), c.b(), c.a()];
                        ts::Color::from_rgba8(r, g, b, a)
                    },
                );

                self.fill_impl(
                    ZenoShape {
                        cmds: outline,
                        offset,
                    },
                    ts::Shader::SolidColor(color),
                    ts::FillRule::EvenOdd,
                );
            } else {
                // Blit the image onto the target.
                let default_color = {
                    let (r, g, b, a) = piet::util::DEFAULT_TEXT_COLOR.as_rgba8();
                    cosmic_text::Color::rgba(r, g, b, a)
                };
                glyph_cache.with_pixels(system, physical.cache_key, default_color, |x, y, clr| {
                    let [r, g, b, a] = [clr.r(), clr.g(), clr.b(), clr.a()];
                    let color = ts::Color::from_rgba8(r, g, b, a);

                    self.fill_impl(
                        kurbo::Rect::from_origin_size((x as f64, y as f64), (1., 1.)),
                        Shader::SolidColor(color),
                        ts::FillRule::EvenOdd,
                    );
                });
            }
        });

        self.cache.glyph_cache = Some(glyph_cache);
    }
}

macro_rules! leap {
    ($this:expr,$e:expr,$msg:literal) => {{
        match ($e) {
            Some(v) => v,
            None => {
                $this.last_error = Err(Pierror::BackendError($msg.into()));
                return;
            }
        }
    }};
}

impl piet::RenderContext for RenderContext<'_, '_> {
    type Brush = Brush;
    type Image = Image;
    type Text = Text;
    type TextLayout = TextLayout;

    fn status(&mut self) -> Result<(), Pierror> {
        mem::replace(&mut self.last_error, Ok(()))
    }

    fn solid_brush(&mut self, color: piet::Color) -> Self::Brush {
        Brush(BrushInner::Solid(color))
    }

    fn gradient(
        &mut self,
        gradient: impl Into<piet::FixedGradient>,
    ) -> Result<Self::Brush, Pierror> {
        Ok(Brush(match gradient.into() {
            piet::FixedGradient::Linear(lin) => BrushInner::LinearGradient(lin),
            piet::FixedGradient::Radial(rad) => BrushInner::RadialGradient(rad),
        }))
    }

    fn clear(&mut self, region: impl Into<Option<kurbo::Rect>>, color: piet::Color) {
        let region = region.into();
        let state = self.states.last().unwrap();

        if region.is_none() || state.clip.is_none() {
            self.target.fill(cvt_color(color));
        } else {
            let region = region.unwrap_or(kurbo::Rect::new(
                0.0,
                0.0,
                self.target.width() as f64,
                self.target.height() as f64,
            ));
            self.ignore_state = true;
            self.fill_impl(
                region,
                Shader::SolidColor(cvt_color(color)),
                tiny_skia::FillRule::Winding,
            );
            // TODO: Preserve this even in a panic.
            self.ignore_state = false;
        }
    }

    fn stroke(&mut self, shape: impl kurbo::Shape, brush: &impl piet::IntoBrush<Self>, width: f64) {
        self.stroke_styled(shape, brush, width, &piet::StrokeStyle::default())
    }

    fn stroke_styled(
        &mut self,
        shape: impl kurbo::Shape,
        brush: &impl piet::IntoBrush<Self>,
        width: f64,
        style: &piet::StrokeStyle,
    ) {
        let mut stroke = ts::Stroke {
            width: width as f32,
            line_cap: match style.line_cap {
                piet::LineCap::Butt => ts::LineCap::Butt,
                piet::LineCap::Round => ts::LineCap::Round,
                piet::LineCap::Square => ts::LineCap::Square,
            },
            dash: if style.dash_pattern.is_empty() {
                None
            } else {
                let dashes = {
                    let mut dashes = mem::take(&mut self.cache.dash_buffer);
                    dashes.clear();
                    dashes.extend(style.dash_pattern.iter().map(|&x| x as f32));
                    dashes
                };

                ts::StrokeDash::new(dashes, style.dash_offset as f32)
            },
            ..Default::default()
        };

        match style.line_join {
            piet::LineJoin::Bevel => stroke.line_join = ts::LineJoin::Bevel,
            piet::LineJoin::Round => stroke.line_join = ts::LineJoin::Round,
            piet::LineJoin::Miter { limit } => {
                stroke.line_join = ts::LineJoin::Miter;
                stroke.miter_limit = limit as f32;
            }
        }

        let shader = leap!(
            self,
            brush.make_brush(self, || shape.bounding_box()).to_shader(),
            "Failed to create shader"
        );

        self.stroke_impl(shape, shader, &stroke);

        // TODO: Add a way to restore dashes to tiny-skia
    }

    fn fill(&mut self, shape: impl kurbo::Shape, brush: &impl piet::IntoBrush<Self>) {
        let shader = leap!(
            self,
            brush.make_brush(self, || shape.bounding_box()).to_shader(),
            "Failed to create shader"
        );
        self.fill_impl(&shape, shader, tiny_skia::FillRule::Winding)
    }

    fn fill_even_odd(&mut self, shape: impl kurbo::Shape, brush: &impl piet::IntoBrush<Self>) {
        let shader = leap!(
            self,
            brush.make_brush(self, || shape.bounding_box()).to_shader(),
            "Failed to create shader"
        );
        self.fill_impl(&shape, shader, tiny_skia::FillRule::EvenOdd)
    }

    fn clip(&mut self, shape: impl kurbo::Shape) {
        let current_state = self.states.last_mut().unwrap();
        let bitmap_scale =
            ts::Transform::from_scale(self.bitmap_scale as f32, self.bitmap_scale as f32);
        let path = {
            let mut builder = self.cache.path_builder.take().unwrap_or_default();
            cvt_shape_to_skia_path(&mut builder, shape, self.tolerance);
            match builder.finish() {
                Some(path) => path,
                None => return,
            }
        };

        match &mut current_state.clip {
            slot @ None => {
                // Create a new clip mask.
                let mut clip = Mask::new(self.target.width(), self.target.height())
                    .expect("Pixmap width/height should be valid clipmask width/height");
                clip.fill_path(&path, tiny_skia::FillRule::EvenOdd, false, bitmap_scale);
                *slot = Some(clip);
            }

            Some(mask) => {
                mask.intersect_path(&path, tiny_skia::FillRule::EvenOdd, false, bitmap_scale);
            }
        }
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.cache.text
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<kurbo::Point>) {
        let pos = pos.into();
        let mut line_processor = piet_cosmic_text::LineProcessor::new();

        for run in layout.0.layout_runs() {
            for glyph in run.glyphs {
                // Process the line state.
                let color = glyph.color_opt.unwrap_or({
                    let piet_color = piet::util::DEFAULT_TEXT_COLOR;
                    let (r, g, b, a) = piet_color.as_rgba8();
                    cosmic_text::Color::rgba(r, g, b, a)
                });
                line_processor.handle_glyph(glyph, run.line_y, color);

                self.draw_glyph(pos, glyph, run.line_y);
            }
        }
    }

    fn save(&mut self) -> Result<(), Pierror> {
        let current_state = self.states.last().unwrap();
        self.states.push(RenderState {
            transform: current_state.transform,
            clip: current_state.clip.clone(),
        });
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Pierror> {
        if self.states.len() == 1 {
            return Err(Pierror::StackUnbalance);
        }

        self.states.pop();
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Pierror> {
        // We don't need to do anything here.
        Ok(())
    }

    fn transform(&mut self, transform: Affine) {
        self.states.last_mut().unwrap().transform *= transform;
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: piet::ImageFormat,
    ) -> Result<Self::Image, Pierror> {
        let data_buffer = match format {
            piet::ImageFormat::RgbaPremul => buf.to_vec(),
            piet::ImageFormat::RgbaSeparate => buf
                .chunks_exact(4)
                .flat_map(|chunk| {
                    let [r, g, b, a]: &[u8; 4] = chunk.try_into().unwrap();
                    let color = tiny_skia::ColorU8::from_rgba(*r, *g, *b, *a);
                    let premul = color.premultiply();
                    [premul.red(), premul.green(), premul.blue(), premul.alpha()]
                })
                .collect(),
            piet::ImageFormat::Rgb => buf
                .chunks_exact(3)
                .flat_map(|chunk| {
                    let [r, g, b]: &[u8; 3] = chunk.try_into().unwrap();
                    [*r, *g, *b, 0xFF]
                })
                .collect(),
            piet::ImageFormat::Grayscale => buf.iter().flat_map(|&v| [v, v, v, 0xFF]).collect(),
            _ => return Err(Pierror::NotSupported),
        };

        // Convert from the data buffer to a pixel buffer.
        let size = tiny_skia::IntSize::from_wh(
            width.try_into().map_err(|_| Pierror::InvalidInput)?,
            height.try_into().map_err(|_| Pierror::InvalidInput)?,
        )
        .ok_or_else(|| Pierror::InvalidInput)?;
        let pixmap =
            tiny_skia::Pixmap::from_vec(data_buffer, size).ok_or_else(|| Pierror::InvalidInput)?;

        Ok(Image(pixmap))
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        dst_rect: impl Into<kurbo::Rect>,
        interp: piet::InterpolationMode,
    ) {
        let bounds = kurbo::Rect::new(0.0, 0.0, image.0.width().into(), image.0.height().into());
        self.draw_image_area(image, bounds, dst_rect, interp);
    }

    fn draw_image_area(
        &mut self,
        image: &Self::Image,
        src_rect: impl Into<kurbo::Rect>,
        dst_rect: impl Into<kurbo::Rect>,
        interp: piet::InterpolationMode,
    ) {
        // Write a transform rule.
        let src_rect = src_rect.into();
        let dst_rect = dst_rect.into();
        let scale_x = dst_rect.width() / src_rect.width();
        let scale_y = dst_rect.height() / src_rect.height();

        let transform = Affine::translate(-src_rect.origin().to_vec2())
            * Affine::translate(dst_rect.origin().to_vec2())
            * Affine::scale_non_uniform(scale_x, scale_y);

        self.fill_impl(
            dst_rect,
            tiny_skia::Pattern::new(
                image.0.as_ref(),
                tiny_skia::SpreadMode::Repeat,
                cvt_filter(interp),
                1.0,
                cvt_affine(transform),
            ),
            tiny_skia::FillRule::Winding,
        )
    }

    fn capture_image_area(
        &mut self,
        src_rect: impl Into<kurbo::Rect>,
    ) -> Result<Self::Image, Pierror> {
        // Get the rectangle making up the image.
        let src_rect = {
            let src_rect = src_rect.into();

            match ts::IntRect::from_xywh(
                (src_rect.x0 * self.bitmap_scale) as i32,
                (src_rect.y0 * self.bitmap_scale) as i32,
                (src_rect.width() * self.bitmap_scale) as u32,
                (src_rect.height() * self.bitmap_scale) as u32,
            ) {
                Some(src_rect) => src_rect,
                None => return Err(Pierror::InvalidInput),
            }
        };

        self.target
            .as_ref()
            .clone_rect(src_rect)
            .ok_or_else(|| Pierror::InvalidInput)
            .map(Image)
    }

    fn blurred_rect(
        &mut self,
        _rect: kurbo::Rect,
        _blur_radius: f64,
        _brush: &impl piet::IntoBrush<Self>,
    ) {
        self.last_error = Err(Pierror::Unimplemented);
    }

    fn current_transform(&self) -> Affine {
        self.states.last().unwrap().transform
    }
}

impl Brush {
    fn to_shader(&self) -> Option<tiny_skia::Shader<'static>> {
        match &self.0 {
            BrushInner::Solid(color) => Some(Shader::SolidColor(cvt_color(*color))),
            BrushInner::LinearGradient(linear) => tiny_skia::LinearGradient::new(
                cvt_point(linear.start),
                cvt_point(linear.end),
                linear
                    .stops
                    .iter()
                    .map(|s| cvt_gradient_stop(s.clone()))
                    .collect(),
                tiny_skia::SpreadMode::Pad,
                tiny_skia::Transform::identity(),
            ),
            BrushInner::RadialGradient(radial) => tiny_skia::RadialGradient::new(
                cvt_point(radial.center + radial.origin_offset),
                cvt_point(radial.center),
                radial.radius as f32,
                radial
                    .stops
                    .iter()
                    .map(|s| cvt_gradient_stop(s.clone()))
                    .collect(),
                tiny_skia::SpreadMode::Pad,
                tiny_skia::Transform::identity(),
            ),
        }
    }
}

impl piet::IntoBrush<RenderContext<'_, '_>> for Brush {
    fn make_brush<'b>(
        &'b self,
        _piet: &mut RenderContext<'_, '_>,
        _bbox: impl FnOnce() -> kurbo::Rect,
    ) -> std::borrow::Cow<'b, Brush> {
        std::borrow::Cow::Borrowed(self)
    }
}

impl piet::Image for Image {
    fn size(&self) -> kurbo::Size {
        kurbo::Size::new(self.0.width() as f64, self.0.height() as f64)
    }
}

impl piet::Text for Text {
    type TextLayout = TextLayout;
    type TextLayoutBuilder = TextLayoutBuilder;

    fn font_family(&mut self, family_name: &str) -> Option<piet::FontFamily> {
        self.0.font_family(family_name)
    }

    fn load_font(&mut self, data: &[u8]) -> Result<piet::FontFamily, Pierror> {
        self.0.load_font(data)
    }

    fn new_text_layout(&mut self, text: impl piet::TextStorage) -> Self::TextLayoutBuilder {
        TextLayoutBuilder(self.0.new_text_layout(text))
    }
}

impl piet::TextLayoutBuilder for TextLayoutBuilder {
    type Out = TextLayout;

    fn max_width(self, width: f64) -> Self {
        Self(self.0.max_width(width))
    }

    fn alignment(self, alignment: piet::TextAlignment) -> Self {
        Self(self.0.alignment(alignment))
    }

    fn default_attribute(self, attribute: impl Into<piet::TextAttribute>) -> Self {
        Self(self.0.default_attribute(attribute))
    }

    fn range_attribute(
        self,
        range: impl std::ops::RangeBounds<usize>,
        attribute: impl Into<piet::TextAttribute>,
    ) -> Self {
        Self(self.0.range_attribute(range, attribute))
    }

    fn build(self) -> Result<Self::Out, Pierror> {
        Ok(TextLayout(self.0.build()?))
    }
}

impl piet::TextLayout for TextLayout {
    fn size(&self) -> kurbo::Size {
        self.0.size()
    }

    fn trailing_whitespace_width(&self) -> f64 {
        self.0.trailing_whitespace_width()
    }

    fn image_bounds(&self) -> kurbo::Rect {
        self.0.image_bounds()
    }

    fn text(&self) -> &str {
        self.0.text()
    }

    fn line_text(&self, line_number: usize) -> Option<&str> {
        self.0.line_text(line_number)
    }

    fn line_metric(&self, line_number: usize) -> Option<piet::LineMetric> {
        self.0.line_metric(line_number)
    }

    fn line_count(&self) -> usize {
        self.0.line_count()
    }

    fn hit_test_point(&self, point: kurbo::Point) -> piet::HitTestPoint {
        self.0.hit_test_point(point)
    }

    fn hit_test_text_position(&self, idx: usize) -> piet::HitTestPosition {
        self.0.hit_test_text_position(idx)
    }
}

struct ZenoShape<'a> {
    cmds: &'a [ZenoCommand],
    offset: kurbo::TranslateScale,
}

impl Shape for ZenoShape<'_> {
    type PathElementsIter<'iter> = ZenoIter<'iter> where Self: 'iter;

    fn path_elements(&self, _tolerance: f64) -> Self::PathElementsIter<'_> {
        ZenoIter {
            inner: self.cmds.iter(),
            offset: self.offset,
        }
    }

    fn area(&self) -> f64 {
        self.to_path(1.0).area()
    }

    fn perimeter(&self, accuracy: f64) -> f64 {
        self.to_path(accuracy).perimeter(accuracy)
    }

    fn winding(&self, pt: kurbo::Point) -> i32 {
        self.to_path(1.0).winding(pt)
    }

    fn bounding_box(&self) -> kurbo::Rect {
        self.to_path(1.0).bounding_box()
    }
}

#[derive(Clone)]
struct ZenoIter<'a> {
    inner: slice::Iter<'a, ZenoCommand>,
    offset: kurbo::TranslateScale,
}

impl ZenoIter<'_> {
    fn leap(&self) -> impl Fn(&ZenoCommand) -> kurbo::PathEl {
        let offset = self.offset;
        move |&cmd| offset * cvt_zeno_command(cmd)
    }
}

impl Iterator for ZenoIter<'_> {
    type Item = kurbo::PathEl;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(self.leap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n).map(self.leap())
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let m = self.leap();
        self.inner.map(m).fold(init, f)
    }
}

fn cvt_zeno_command(cmd: ZenoCommand) -> kurbo::PathEl {
    let cvt_vector = |v: zeno::Vector| {
        let [x, y]: [f32; 2] = v.into();
        kurbo::Point::new(x as f64, y as f64)
    };

    match cmd {
        ZenoCommand::Close => kurbo::PathEl::ClosePath,
        ZenoCommand::MoveTo(p) => kurbo::PathEl::MoveTo(cvt_vector(p)),
        ZenoCommand::LineTo(p) => kurbo::PathEl::LineTo(cvt_vector(p)),
        ZenoCommand::QuadTo(p1, p2) => kurbo::PathEl::QuadTo(cvt_vector(p1), cvt_vector(p2)),
        ZenoCommand::CurveTo(p1, p2, p3) => {
            kurbo::PathEl::CurveTo(cvt_vector(p1), cvt_vector(p2), cvt_vector(p3))
        }
    }
}

fn cvt_shape_to_skia_path(builder: &mut PathBuilder, shape: impl Shape, tolerance: f64) {
    shape.path_elements(tolerance).for_each(|el| match el {
        PathEl::ClosePath => builder.close(),
        PathEl::MoveTo(p) => builder.move_to(p.x as f32, p.y as f32),
        PathEl::LineTo(p) => builder.line_to(p.x as f32, p.y as f32),
        PathEl::QuadTo(p1, p2) => {
            builder.quad_to(p1.x as f32, p1.y as f32, p2.x as f32, p2.y as f32)
        }
        PathEl::CurveTo(p1, p2, p3) => builder.cubic_to(
            p1.x as f32,
            p1.y as f32,
            p2.x as f32,
            p2.y as f32,
            p3.x as f32,
            p3.y as f32,
        ),
    })
}

fn cvt_affine(p: kurbo::Affine) -> tiny_skia::Transform {
    let [a, b, c, d, e, f] = p.as_coeffs();
    tiny_skia::Transform::from_row(a as f32, b as f32, c as f32, d as f32, e as f32, f as f32)
}

fn cvt_gradient_stop(stop: piet::GradientStop) -> tiny_skia::GradientStop {
    tiny_skia::GradientStop::new(stop.pos, cvt_color(stop.color))
}

fn cvt_color(p: piet::Color) -> tiny_skia::Color {
    let (r, g, b, a) = p.as_rgba();
    tiny_skia::Color::from_rgba(r as f32, g as f32, b as f32, a as f32).expect("Color out of range")
}

fn cvt_point(p: kurbo::Point) -> tiny_skia::Point {
    tiny_skia::Point {
        x: p.x as f32,
        y: p.y as f32,
    }
}

fn cvt_filter(p: piet::InterpolationMode) -> tiny_skia::FilterQuality {
    match p {
        piet::InterpolationMode::NearestNeighbor => tiny_skia::FilterQuality::Nearest,
        piet::InterpolationMode::Bilinear => tiny_skia::FilterQuality::Bilinear,
    }
}
