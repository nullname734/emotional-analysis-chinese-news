# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ScNewsItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field() #标题
    sourse = scrapy.Field() #来源
    content = scrapy.Field() #内容
    date = scrapy.Field() #日期
    ip = scrapy.Field() #发布ip
    pass
