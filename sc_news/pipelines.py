# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import sqlite3
class Sqlite3Pipeline(object):
    def open_spider(self,spider):
        print('导入数据中……')
        self.conn = sqlite3.connect('D:\\Users\\yang2\\Desktop\\毕业设计\\code\\sc_news\\news.db')
        self.cur = self.conn.cursor()
        #不存在数据表则需要创建
        self.cur.execute('''
        create table if not exists today_news(
        id integer primary key autoincrement,
        title text not NULL,
        the_source text not NULL,
        content text not NULL,
        pub_date date not NULL,
        ip text not NULL
        )''')

    def process_item(self,item,spider):
        insert_sql ="insert into today_news(title,the_source,content,pub_date, ip)" \
                    "values(?,?,?,?,?)" #添加数据

        #print('添加数据：')
        data = (item['title'], item['sourse'], item['content'], item['date'], item['ip'])

        self.cur.execute(insert_sql,data)
        self.conn.commit()

        return item

    def close_spider(self,spider):
        print('完成导入')
        self.cur.close()
        self.conn.close()