import re
import scrapy
from selenium import webdriver  #web应用程序测试，运行在浏览器中
from selenium.webdriver.chrome.service import Service
from sc_news.items import ScNewsItem


class NewsSpider(scrapy.Spider):
    name = "news"
    start_urls = ["http://news.163.com/"]
    model_urls = []  #存储板块对应详情页的url

    #实例化一个浏览器对象
    def __init__(self):
        super().__init__()
        service = Service(
            executable_path='D:\\Users\\yang2\\Desktop\\毕业设计\\chrome-win64\\chrome-win64\\chromedriver.exe')
        self.bro = webdriver.Chrome(service=service)  #谷歌浏览器


    def parse(self, response):
        #因为节日广告等原因必须用相对路径解析
        li_list = response.xpath('//*[@id="index2016_wrap"]//div[@class="ns_area list"]/ul/li')  #首页栏
        model_list = [1, 2, 4, 5]

        for index in model_list:
            model_url = li_list[index].xpath('./a/@href').extract_first()
            self.model_urls.append(model_url)
        #对各个板块页面进行请求
        for url in self.model_urls:  #发送请求
            yield scrapy.Request(url=url, callback=self.parse_model)

    # 对应新闻标题相关的内容是动态加载
    def parse_model(self, response):
        #print('访问新闻版面')
        #所有的新闻的位置都相同，除了航空
        div_list = response.xpath(
            '//*[@class="ns9"]//div[@class="ns_area second2016_main clearfix"]//div[@class="ndi_main"]/div')
        for div in div_list:
            title = div.xpath('.//div[@class="news_title"]/h3/a/text()').extract_first()
            new_detail_url = div.xpath('.//div[@class="news_title"]/h3/a/@href').extract_first()
            item = ScNewsItem()  #实例化一个item对象
            item['title'] = title  #把抓取到的title放入对象中

            #对新闻详情页的url发起请求
            if title != None:  #在新闻路径下有部分广告div，需要跳过
                yield scrapy.Request(url=new_detail_url, callback=self.parse_detail, meta={'item': item})

    @classmethod
    def parse_detail(self, response):
        #print('访问新闻详情页')
        sourse = response.xpath('//div[@class="post_info"]/a/text()').extract_first()  #新闻来源
        date = response.xpath('//div[@class="post_info"]/text()').getall()  # 新闻日期
        content = response.xpath('//div[@class="post_body"]//p/text()').getall()  #新闻内容
        content = '\n'.join([p.strip() for p in content if p.strip()])
        #print(date[0])
        re_date = re.search(r'\d{4}-\d{1,2}', date[0]).group()
        place_ip = date[2].replace(' ', '').replace('\n', '').replace('\xa0', '')
        print(place_ip)
        #print(re_date)
        item = response.meta['item']
        item['sourse'] = sourse
        item['content'] = content
        item['date'] = re_date
        item['ip'] = place_ip
        #提交到数据库
        yield item

    def closed(self, spider):
        self.bro.quit()  #关闭浏览器
