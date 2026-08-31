source "https://rubygems.org"

# The full "github-pages" meta-gem can't install on Ruby 4.x: it pulls in
# jekyll-commonmark-ghpages -> commonmarker, a native extension capped at
# Ruby < 4.0. This site renders markdown with kramdown (see _config.yml),
# so commonmarker is never actually used — declare Jekyll and only the
# plugins _config.yml enables directly, pinned to the exact versions
# GitHub Pages currently serves (github-pages gem 232) so local output
# matches production.
gem "jekyll", "3.10.0"
gem "liquid", "4.0.4" # 4.0.3 (Jekyll's default pin target) calls String#tainted?, removed in Ruby 3.2+
gem "jekyll-feed", "0.17.0"
gem "jekyll-seo-tag", "2.8.0"
gem "jekyll-sitemap", "1.4.0"
gem "jekyll-redirect-from", "0.16.0"
gem "kramdown-parser-gfm", "1.1.0"
gem "webrick", "~> 1.8"

# Ruby 3.4+ no longer bundles these with the interpreter; Jekyll 3.x
# still requires them at runtime.
gem "csv"
gem "logger"
gem "base64"
gem "bigdecimal"
