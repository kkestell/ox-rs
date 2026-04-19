use axum::body::Body;
use axum::http::header;
use axum::response::Response;

use crate::assets;

pub(super) async fn index_html() -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(assets::INDEX_HTML))
        .expect("building index response")
}

pub(super) async fn app_js() -> Response {
    Response::builder()
        .header(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(assets::APP_JS))
        .expect("building app.js response")
}

pub(super) async fn styles_css() -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/css; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(assets::STYLES_CSS))
        .expect("building styles.css response")
}
