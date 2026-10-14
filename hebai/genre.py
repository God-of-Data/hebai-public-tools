"""URL -> genre for web text (FineWeb-2 / mC4). Rules only, so every assignment can be explained.

Genres (product-relevant, decided 2026-10-13):
  personal   blogs, personal sites, diaries
  forum      forums and Q&A sites (questions, replies)
  comments   talkbacks / comment sections (rare on crawled pages; mostly from the comment corpora)
  news       news and magazine articles, op-eds
  business   company / shop / service pages, marketing copy
  formal     government, municipalities, universities, NGOs, legal and institutional text
  howto      guides, recipes, tips, health information
  encyclo    Wikipedia and other encyclopedic sites
  other      anything not matched (kept for FPR measurement, not for genre quotas)
"""

from __future__ import annotations

import re
import urllib.parse

NEWS = {
    "ynet.co.il", "walla.co.il", "maariv.co.il", "haaretz.co.il", "mako.co.il", "nrg.co.il", "news1.co.il",
    "kikar.co.il", "bhol.co.il", "inn.co.il", "calcalist.co.il", "globes.co.il", "themarker.com", "israelhayom.co.il",
    "0404.co.il", "srugim.co.il", "davar1.co.il", "ice.co.il", "bizportal.co.il", "sport5.co.il", "one.co.il",
    "news.walla.co.il", "kan.org.il", "13tv.co.il", "reshet.tv", "i24news.tv", "jdn.co.il", "actualic.co.il",
    "hamodia.co.il", "makorrishon.co.il", "mida.org.il", "zman.co.il", "local.co.il", "mynet.co.il", "kipa.co.il",
    "ice.co.il", "funder.co.il", "ynetnews.com", "nana10.co.il", "news08.net", "hm-news.co.il", "behadrei.co.il",
    "doctorsonly.co.il", "geektime.co.il", "pc.co.il", "people-and-computers.co.il", "the7eye.org.il", "sponser.co.il",
}
FORUM = {
    "tapuz.co.il", "fxp.co.il", "rotter.net", "hydepark.co.il", "stips.co.il", "askpeople.co.il", "askp.co.il",
    "forum.kipa.co.il", "forums.walla.co.il", "kipa.co.il/forum", "bhol.co.il/forums", "prog.co.il", "hwzone.co.il",
    "whatsup.co.il", "pirsumim.co.il", "lo-tov.co.il", "ladies.co.il", "family.co.il",
}
BLOG = {"israblog.nana10.co.il", "blogspot.com", "wordpress.com", "blogs.timesofisrael.com", "cafe.themarker.com",
        "blogli.co.il", "blog.tapuz.co.il", "sheee.co.il", "medium.com"}
ENCYCLO = {"wikipedia.org", "he.wikipedia.org", "he.wikibooks.org", "he.wikiquote.org", "he.wikisource.org",
           "wikiwand.com", "hamichlol.org.il", "daat.ac.il", "snunit.k12.il"}
HOWTO = {"kolzchut.org.il", "infomed.co.il", "doctors.co.il", "mouse.co.il", "foodsdictionary.co.il", "10dakot.co.il",
         "ptitim.co.il", "nikuyit.net", "mako.co.il/food", "food.walla.co.il", "xnet.ynet.co.il", "tapuz.co.il/blogs",
         "health.walla.co.il", "betipul.co.il", "hidabroot.org", "parents.co.il", "mamy.co.il", "babyworld.co.il"}
LEGAL = {"nevo.co.il", "psakdin.co.il", "takdin.co.il", "gov.il", "court.gov.il", "knesset.gov.il", "justice.gov.il",
         "lawdata.co.il", "kolzchut.org.il/he/"}

_PATH_HOWTO = re.compile(r"(recipe|matkon|%D7%9E%D7%AA%D7%9B%D7%95%D7%9F|/tips?/|/guide|/how-?to|/madrich)", re.I)
_PATH_BLOG = re.compile(r"(/blog|/posts?/|/\d{4}/\d{2}/)", re.I)
_PATH_SHOP = re.compile(r"(/product|/shop|/store|/catalog|/cart|/item|/p/\d|/מוצר)", re.I)
_PATH_FORUM = re.compile(r"(/forum|/threads?/|/topic|/viewtopic|showthread|/question)", re.I)
_PATH_NEWS = re.compile(r"(/article|/news|/item/\d|/Article-|/opinion)", re.I)


def _match(host: str, path: str, table: set) -> bool:
    for d in table:
        if "/" in d:
            h, p = d.split("/", 1)
            if (host == h or host.endswith("." + h)) and path.lstrip("/").startswith(p):
                return True
        elif host == d or host.endswith("." + d):
            return True
    return False


def host_of(url: str) -> str:
    h = urllib.parse.urlparse(url or "").netloc.lower()
    h = h.split("@")[-1].split(":")[0]
    return h[4:] if h.startswith("www.") else h


# ---- v2 (2026-10-13, after the 50-per-genre audit) -------------------------------------------------------
# Audit findings: machine-translated spam sites (iw.*/he.* subdomains of foreign sites, /iw/ paths, junk TLDs) are
# machine text and must never be "human"; classical / old-orthography sources (Wikisource, Daat classics, Sefaria,
# Ben-Yehuda, Chabad library) repeat the Ben-Yehuda register confound; the path rules and the .co.il/.com catch-all
# gave low precision. v2 = exclusions + host lists + a few high-precision path/text rules; everything else "other".
OK_FOREIGN_TLD = {"com", "org", "net", "co", "me", "io", "blog", "tv", "fm", "news"}
OLD_REGISTER = {"wikisource.org", "daat.ac.il", "sefaria.org", "benyehuda.org", "chabadlibrary.org", "kab.li",
                "hebrewbooks.org", "otzar.org", "toldot.com", "mechon-mamre.org", "on-line.co.il/ok"}
NEWS2 = NEWS | {"colbonews.co.il", "kolhazman.co.il", "kolhair.co.il", "20il.co.il", "chasidinews.com", "shturem.net",
                "ladaat.co", "mashabim.news", "news-israel.net", "yavnenet.co.il", "kolhair-modiin.co.il", "96fm.co.il",
                "arimnews.co.il", "talniri.co.il/marketnews", "nadlancenter.co.il", "ice.co.il", "mivzaklive.co.il",
                "israeldefense.co.il", "frogi.co.il", "sport.nana10.co.il", "bidur.nana10.co.il", "10tv.nana10.co.il",
                "safsal.co.il", "kikar.co.il", "emess.co.il", "hidabroot.org/article", "haaretz.co.il", "themarker.com",
                "calcalist.co.il", "globes.co.il", "maariv.co.il", "walla.co.il", "pc.co.il", "zman.co.il",
                "the7eye.org.il", "makorrishon.co.il", "davar1.co.il", "inn.co.il", "srugim.co.il", "bhol.co.il"}
NEWS2 -= {"kipa.co.il/forum"}
FORUM_PATH = re.compile(r"(viewtopic|showthread|forum_showMsg|forumdisplay|/forums?/|/threads/|stips\.co\.il/ask|"
                        r"/question/|/msg_\d|forum_msg|forumIn\.asp|/topic/\d)", re.I)
NOT_FORUM = re.compile(r"(rotter\.net/(?:blogs|forum/scoops)|hwzone\.co\.il/main-|davar1\.co\.il/topic|"
                       r"hayadan\.org\.il/topic|safariyam)", re.I)
BLOG_HOSTS = {"blogspot.com", "blogspot.co.il", "wordpress.com", "israblog.nana10.co.il", "cafe.themarker.com",
              "blogli.co.il", "blog.tapuz.co.il", "sheee.co.il", "medium.com", "blogs.timesofisrael.com", "blogim.co.il",
              "tapuz.co.il/blogs"}
FIRST_PERSON = re.compile(r"(?<![א-ת])(?:אני|שלי|הייתי|אצלי|לי|אותי|עליי|עלי|חשבתי|הרגשתי|נזכרתי|ראיתי|הלכתי)(?![א-ת])")
RECIPE_PATH = re.compile(r"(recipe|/matkon|מתכון|/recipes/)", re.I)
HOWTO_HOSTS = {"kolzchut.org.il", "infomed.co.il", "foodsdictionary.co.il", "10dakot.co.il", "ptitim.co.il",
               "food.walla.co.il", "foodisgood.co.il", "cookshare.co.il", "lametayel.co.il/tips", "midrag.co.il/Content/Tip",
               "mamy.co.il", "babyworld.co.il", "parents.co.il", "erez-komarovsky.co.il", "shukhanamal.co.il"}
SHOP_PATH = re.compile(r"(/product[s/-]|/shop/|/store/|/catalog/|/cart|/product-page/|/items?/\d|/מוצר)", re.I)
MARKETING = re.compile(r"(אנו מציעים|אנחנו מציעים|השירותים שלנו|לקוחותינו|הלקוחות שלנו|החברה שלנו|אנו מתמחים|אנחנו מתמחים|"
                       r"צוות המומחים|ניסיון של למעלה|מגוון רחב של|במחירים|השירות שלנו|פנו אלינו|התקשרו עוד היום|"
                       r"לפרטים נוספים|מומחים בתחום)")


def exclude_reason(url: str):
    host = host_of(url)
    path = urllib.parse.urlparse(url or "").path
    if any(host == d or host.endswith("." + d) for d in OLD_REGISTER if "/" not in d):
        return "old_register"
    if host.endswith(".il"):
        return None
    tld = host.rsplit(".", 1)[-1]
    if host.startswith(("iw.", "he.")) and not re.search(r"wiki(pedia|books|quote|news|tionary)\.org$", host):
        return "machine_translated"
    if re.search(r"/(?:iw|he)(?:[-_]il)?/", path, re.I) and not host.endswith("wikipedia.org"):
        return "machine_translated"
    if tld not in OK_FOREIGN_TLD:
        return "foreign_tld"
    return None


def genre_v2(url: str, text: str = "") -> str:
    u = urllib.parse.urlparse(url or "")
    host, path = host_of(url), urllib.parse.unquote(u.path or "") + ("?" + u.query if u.query else "")
    full = host + path
    if _match(host, path, ENCYCLO - {"he.wikisource.org", "daat.ac.il", "snunit.k12.il"}):
        return "encyclo"
    if NOT_FORUM.search(full):
        return "other"
    if _match(host, path, FORUM) or FORUM_PATH.search(full):
        return "forum"
    if _match(host, path, NEWS2):
        return "news"
    if _match(host, path, BLOG_HOSTS) or host.startswith("blog."):
        return "personal" if len(FIRST_PERSON.findall(text)) >= 2 else "other"
    if _match(host, path, HOWTO_HOSTS) or RECIPE_PATH.search(path):
        return "howto"
    if host.endswith((".gov.il", ".ac.il", ".muni.il", ".k12.il", ".idf.il")) or _match(host, path, LEGAL):
        return "formal"
    if _PATH_BLOG.search(path) and len(FIRST_PERSON.findall(text)) >= 3:
        return "personal"
    if SHOP_PATH.search(path) or (MARKETING.search(text) and (host.endswith(".co.il") or host.endswith(".com"))):
        return "business"
    return "other"


def genre_of(url: str) -> str:
    u = urllib.parse.urlparse(url or "")
    host, path = host_of(url), urllib.parse.unquote(u.path or "") + ("?" + u.query if u.query else "")
    if _match(host, path, ENCYCLO):
        return "encyclo"
    if _match(host, path, FORUM) or _PATH_FORUM.search(path):
        return "forum"
    if _match(host, path, BLOG) or host.startswith("blog.") or ".blog." in host:
        return "personal"
    if _match(host, path, HOWTO) or _PATH_HOWTO.search(path):
        return "howto"
    if _match(host, path, LEGAL) or host.endswith((".gov.il", ".ac.il", ".muni.il", ".k12.il", ".idf.il")) or host.endswith(".org.il"):
        return "formal"
    if _match(host, path, NEWS) or _PATH_NEWS.search(path):
        return "news"
    if _PATH_BLOG.search(path):
        return "personal"
    if _PATH_SHOP.search(path) or host.endswith(".co.il") or host.endswith(".com"):
        return "business"
    return "other"
