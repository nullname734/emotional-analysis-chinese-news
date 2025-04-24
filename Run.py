import sys
import os
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from pathlib import Path
import subprocess
import sqlite3

if __name__ == '__main__':
    #清理前日新闻
    con = sqlite3.connect('D:\\Users\\yang2\\Desktop\\毕业设计\\code\\sc_news\\news.db')
    cursor = con.cursor()
    cursor.execute('drop table if exists today_news')
    # 提交更改，关闭连接
    con.commit()
    con.close()

    #爬虫获取今日新闻
    scrapy_path = os.path.join(os.path.dirname(__file__), 'sc_news')
    sys.path.insert(0, scrapy_path)
    #环境变量
    os.environ['SCRAPY_SETTINGS_MODULE'] = 'sc_news.settings'
    settings = get_project_settings()
    #创建实例
    from sc_news.spiders.news import NewsSpider
    process = CrawlerProcess(settings)
    process.crawl(NewsSpider)
    process.start()

    #运行网页
    current_dir = Path(__file__).parent
    django_dir = current_dir / 'Web'
    os.chdir(django_dir)
    try:
        subprocess.run(['python', 'manage.py', 'runserver'], check=True)
    except subprocess.CalledProcessError as e:
        print(e)