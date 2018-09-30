#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
from functools import partial


PATH = 'content'

# Blog Conf
AUTHOR = 'Lee-W'
SITENAME = 'Life Lies in Traveling'
SITEURL = 'http://localhost:8000'
SITETITLE = AUTHOR
SITELOGO = '//s.gravatar.com/avatar/3e049ed33195d00a8b745b16c63dce6e?s=120'
TIMEZONE = 'Asia/Taipei'
DEFAULT_LANG = 'zh'
DEFAULT_CATEGORY = 'Article'
BROWSER_COLOR = '#333333'
DISQUS_SITENAME = "lee-w-blog"
MAIN_MENU = True

MENUITEMS = (
    ('Archives', '/archives.html'),
    ('Categories', '/categories.html'),
    ('Tags', '/tags.html'),
)

DATE_FORMATS = {
    'zh': '%Y/%m/%d - %a',
}

TYPOGRIFY = False
ARTICLE_URL = 'posts/{category}/{date:%Y}/{date:%m}/{slug}'
ARTICLE_SAVE_AS = 'posts/{category}/{date:%Y}/{date:%m}/{slug}/index.html'
DEFAULT_PAGINATION = 10

STATIC_PATHS = ['extra', 'images', 'static']

# Theme Setting
THEME = 'theme/pelican-clean-blog/'
PYGMENTS_STYLE = 'xcode'
EXTRA_PATH_METADATA = {
    'images': {'path': 'images'},
}
CSS_OVERRIDE = '/static/custom.css'
HEADER_COVER = '/images/cover.jpg'

# Feed generation is usually not desired when developing

FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Social widget
SOCIAL = (('linkedin', 'http://tw.linkedin.com/in/clleew'),
          ('github', 'http://github.com/Lee-W'),
          ('gitlab', 'https://gitlab.com/Lee-W'),
          ('facebook', 'http://www.facebook.com/clleew'),
          ('rss', '//lee-w.github.io/feeds/all.atom.xml'),)


# Markdown extension
MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.extra': {},
        'markdown.extensions.codehilite': {},
        'markdown.extensions.nl2br': {},
        'del_ins': {},
    },
    'output_format': 'html5'
}

# Plugin-setting
PLUGIN_PATHS = ['pelican-plugins']
PLUGINS = ['another_read_more_link', 'series', 'render_math', 'neighbors', 'share_post']
ANOTHER_READ_MORE_LINK = ''

# Custom Jinja Filters
JINJA_FILTERS = {
    'sort_by_article_count': partial(
        sorted,
        key=lambda x: len(x[1]),
        reverse=True
    )
}
