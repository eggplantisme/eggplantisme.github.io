---
layout: post
title: 从Request到Selenium的简单爬虫学习笔记
tags: [Learn]
date: 2022-02-22 18:46
---


### 从Request到Selenium的简单爬虫学习笔记

selenium是一个用于网站测试的工具，是一个Web自动化工具，测试人员必会的工具。他可以操作浏览器对网页进行模拟人的操作，比如点击，获取文本数据，跳转等等。所以也可以被用来爬虫。

简单的网站爬虫用request就可以实现，但由于反爬虫技术的出现，对于一些网站使用request就需要更多的技巧去爬数据，而且现在大多数的网站采用js渲染网页的技术，直接用request获取可能得到的并不是浏览器渲染的网页，而是一堆带有js代码的html文件。所以使用selenium操纵浏览器去访问这样的网站就比单单用request简单许多，浏览器帮助我们解决了很大部分问题，但是随之而来的就是效率会很慢。浏览器打开网页会把所有资源都加载进来，包括图片，css，js。不如request直接加载html就能获取到想要的数据那么快。如果技术够强，并且想要的数据比较大， 还是使用request效率更高。

#####  下载图片

- request

  [request文档](https://requests.readthedocs.io/en/master/)

  [Beautifulsoup文档](https://www.crummy.com/software/BeautifulSoup/bs4/doc/ )

  我们下载煎蛋网的meizi图片，网址url=http://jandan.net/ooxx。

  首先使用request.get(url)访问网页，然后使用Beautifusoup解析网页，获取网页你想要的图片的链接地址（需要通过chrome开发工具F12来了解网页结构，然后编写代码提取网页的元素。）。

  然后就是下载图片，为了在下载过程中使用进度条展示下载进度，可以使用流的方式下载。

  最后就是翻页，网站不可能一页就展示所有内容，大多数时刻是需要分页的，我们找完一页的数据之后，可以看看是否有下一页，如果有我们就跳转到下一页的链接，否则就结束爬虫。

  ``` python
  import requests
  import os
  from contextlib import closing
  from bs4 import BeautifulSoup
  
  
  def img_download(folder, img_href, headers):
      if not os.path.exists(folder):
          os.mkdir(folder)
      for src in img_href:
          # 下载时显示进度条
          with closing(requests.get("http:" + src, headers=headers, stream=True)) as r:
              chunk_size = 1024
              content_size = int(r.headers['content-length'])
              file_name = src.split('/')[-1]
              progress = ProgressBar(file_name, total=content_size, unit="KB", chunk_size=chunk_size, run_status="正在下载", fin_status="下载完成")
              print(r.status_code, src)
              with open(folder + file_name, 'wb') as f:
                  for data in r.iter_content(chunk_size=chunk_size):
                      f.write(data)
                      progress.refresh(count=len(data))
  
  
  class ProgressBar(object):
      def __init__(self, title, count=0.0, run_status=None, fin_status=None, total=100.0, unit='', sep='/', chunk_size=1.0):
          super(ProgressBar, self).__init__()
          self.info = "【%s】%s %.2f %s %s %.2f %s "
          self.title = title
          self.total = total
          self.count = count
          self.chunk_size = chunk_size
          self.status = run_status or ""
          self.fin_status = fin_status or " " * len(self.status)
          self.unit = unit
          self.seq = sep
  
      def __get_info(self):
          # 【名称】状态 进度 单位 分割线 总数 单位
          _info = self.info % (self.title, self.status, self.count/self.chunk_size, self.unit, self.seq, self.total/self.chunk_size, self.unit)
          return _info
  
      def refresh(self, count=1, status=None):
          self.count += count
          self.status = status or self.status
          end_str = "\r"
          percent = self.count / self.total
          bar = '*' * int(10 * percent) + '-' * (10 - int(10 * percent))
          if self.count >= self.total:
              end_str = '\n'
              self.status = status or self.fin_status
              bar = '*' * 10
          print(self.__get_info() + bar, end=end_str)
  
          
  def main():
      headers = {
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
          'Accept-Encoding': 'gzip, deflate',
          'Accept-Language': 'zh-CN,zh;q=0.9',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Upgrade-Insecure-Requests': '1',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'
      }
      url = 'http://jandan.net/ooxx'
      while url is not None:
          # 请求
          r = requests.get(url, headers=headers)
          html = r.text
          # 解析
          soup = BeautifulSoup(html, 'lxml')
          img_div = soup.find_all(class_='text')
          img_href = []
          for d in img_div:
              img = d.find('a', class_='view_img_link')
              img_href.append(img['href'])
          cur = soup.find('span', class_='current-comment-page')
          print('page', cur.text, r.status_code, url)
          # print(img_href)
          # 下载
          folder = "./img/meizi/"
          img_download(folder, img_href, headers)
          # 下一页
          next_page = soup.find('a', class_='previous-comment-page')
          if next_page is None:
              url = None
          else:
              url = 'http:' + next_page['href']
  
  
  if __name__ == '__main__':
      main()
  
  ```

- selenium

  [Selenium文档](https://selenium-python.readthedocs.io/index.html)

  首先需要安装selenium库，然后从[http://chromedriver.storage.googleapis.com/index.html](http://chromedriver.storage.googleapis.com/index.html) 下载chromedriver，解压到符合你文件管理的路径下，然后将路径添加到环境变量中(Windows下), 在cmd或powershell输入chromedriver如果没有错误就可以了。就可以写代码了。

  我们从host = [https://bing.ioliu.cn/](https://bing.ioliu.cn/)下载bing的壁纸。通过

  ```python
  from selenium import webdriver
  profile = webdriver.ChromeOptions()
  driver = webdriver.Chrome(chrome_options=profile)
  driver.get(host)
  ```

  打开chrome浏览器并访问网站。其中

  ```python
  profile = webdriver.ChromeOptions()
  profile.add_experimental_option("prefs", {"download.default_directory": "D:\\Code\\python\\Spider\\img\\bingwallpaper"})
  ```

  是设置打开的浏览器的默认保存位置。

  然后我们观察到这个网页上的图片上有下载按钮，所以我们直接操纵浏览器点击下载按钮即可，但是为了防止下载过于频繁，我们每次点击下载后会暂停几秒。driver通过`find_elements_by_class_name`函数按照class搜寻元素，元素的`get_attribute`函数可以获取属性信息。

  最后点击下一页。点击下载按钮和点击下一页都需要用到selenium的ActionChains，比如点击下一页

  ```python
  from selenium.webdriver.common.action_chains import ActionChains
  next = driver.find_element_by_link_text("下一页")
  ActionChains(driver).click(next).perform()
  ```

  完整代码

  ``` python
  import time
  import random
  from selenium import webdriver
  from selenium.webdriver.common.action_chains import ActionChains
  from selenium.common.exceptions import NoSuchElementException
  
  
  def selenium_main():
      host = "https://bing.ioliu.cn/"
      profile = webdriver.ChromeOptions()
      profile.add_experimental_option("prefs", {"download.default_directory": "D:\\Code\\python\\Spider\\img\\bingwallpaper"})
      driver = webdriver.Chrome(chrome_options=profile)
      driver.implicitly_wait(10)
      driver.get(host)
      next = [1]
      try:
          while next is not [] or next is not None:
              img_dls = driver.find_elements_by_class_name('download')
              srcs = []
              alts = []
              for img in img_dls:
                  src = img.get_attribute("href")
                  print("src: " + src)
                  srcs.append(src)  # url
                  ActionChains(driver).click(img).perform()
                  time.sleep(random.randint(3, 5))
              next = driver.find_element_by_link_text("下一页")
              ActionChains(driver).click(next).perform()
              time.sleep(3)
          input()
  
      except NoSuchElementException as e:
          print(e)
      finally:
          driver.close()
  
  
  if __name__ == '__main__':
     # main()  # 会被封IP
      selenium_main()
  ```

##### 知乎用户关系(爬取失败)

爬取知乎用户关系和上面逻辑差不多，但是知乎这样的网站肯定有很多办法应对爬虫。首先对知乎用户界面一通分析观察，我们开始写代码，获取用户的一些信息，然后遍历用户的关注和被关注列表，然后将数据以一个自定义的User数据结构储存。

但是知乎会识别你的浏览器是否是被自动测试软件控制的，所以这里还需要一些别的方式，参照[这里](https://www.cnblogs.com/future-dream/p/11109124.html)，我们需要打开系统的浏览器然后用selenium监控我们打开的浏览器。

```python
options = webdriver.ChromeOptions()
# 需要将系统的chrome.exe加入环境变量，并且执行
# chrome.exe --remote-debugging-port=9222 --user-data-dir="D:\Code\python\Spider\selenium-portal"
# 此时打开系统的chrome,之后用selenium接管浏览器，不被网站识别。
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")  # 接管本机的chrome
self.driver = webdriver.Chrome(chrome_options=options)
```

如果没有什么意外，我们用DFS或BFS都可以搜索用户关系信息，然后构造一个网络进行分析，但意外是会发生的，当你频繁点击知乎时，知乎会识别出来并给你一个验证码让你填。解决方法可以使用机器学习的方法自动识别验证码（还未实现）。目前的代码如下

```python
import time
import json
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException


class User:
    iid = 0

    def __init__(self, url_id, _type, name='', agreed_num=0, following_ids=[], follower_ids=[]):
        self.id = User.iid
        User.iid += 1
        self.url_id = url_id  # id
        self.name = name
        self.type = _type  # 类型
        self.agreed_num = agreed_num
        self.following_ids = following_ids  # 关注的user的id
        self.follower_ids = follower_ids  # 被关注的user的id

    def __str__(self):
        return 'id:' + str(self.id) + '\t' + str(self.url_id) + '\t' + str(self.type) + '\t' + str(self.agreed_num)


class ZhihuUsership:

    def __init__(self):
        self.url_ids = set()
        options = webdriver.ChromeOptions()
        # 需要将系统的chrome.exe加入环境变量，并且执行
        # chrome.exe --remote-debugging-port=9222 --user-data-dir="D:\Code\python\Spider\selenium-portal"
        # 此时打开系统的chrome,之后用selenium接管浏览器，不被网站识别。
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")  # 接管本机的chrome
        self.driver = webdriver.Chrome(chrome_options=options)

    def save(self, root_user, file_name):
        root_user_json = {"id": str(root_user.id), "url_id": str(root_user.url_id), "name": root_user.name,
                          "type": root_user.type, "agree_num": str(root_user.agreed_num),
                          "following_ids": root_user.following_ids,
                          "follower_ids": root_user.follower_ids}
        with open(file_name, "r") as fr:
            before_json = json.load(fr)
            before_json.append(root_user_json)
        with open(file_name, "w") as fw:
            json.dump(before_json, fw)

    def login(self):
        print("Login...(not implemented)")
        pass

    def get_follow(self, root_user, following_url, follower_url):
        # 遍历following
        followings = []  # 关注的User列表
        following_as = self.driver.find_elements_by_class_name('UserLink-link')  # 关注的人主页链接元素列表
        following_as = [following_as[i] for i in range(len(following_as)) if i % 2 == 0]  # 同一个有头像和名字两个链接，取一个
        # 遍历关注列表里的分页
        while True:
            # 处理链接列表然后生成user列表
            for following_a in following_as:
                href = following_a.get_attribute('href')
                _type = href.split('/')[3]
                _url_id = href.split('/')[4]
                followings.append(User(url_id=_url_id, _type=_type))
            # 点击下一页
            next_button = self.driver.find_elements_by_class_name('PaginationButton-next')
            if next_button == []:
                break
            next_button = next_button[0]
            ActionChains(self.driver).click(next_button).perform()
            time.sleep(3)
            following_as = self.driver.find_elements_by_class_name('UserLink-link')  # 关注的人主页链接元素列表
            following_as = [following_as[i] for i in range(len(following_as)) if i % 2 == 0]  # 同一个有头像和名字两个链接，取一个
        print(root_user, " following number: ", len(followings))

        # 遍历followers
        self.driver.get(follower_url)
        followers = []  # 被关注的User列表
        follower_as = self.driver.find_elements_by_class_name('UserLink-link')  # 被关注的人主页链接元素列表
        follower_as = [follower_as[i] for i in range(len(follower_as)) if i % 2 == 0]
        # 遍历关注列表里的分页
        while True:
            # 处理链接列表然后生成user列表
            for follower_a in follower_as:
                href = follower_a.get_attribute('href')
                _type = href.split('/')[3]
                _url_id = href.split('/')[4]
                followers.append(User(url_id=_url_id, _type=_type))
            # 点击下一页
            next_button = self.driver.find_elements_by_class_name('PaginationButton-next')
            if next_button == []:
                break
            next_button = next_button[0]
            ActionChains(self.driver).click(next_button).perform()
            time.sleep(3)
            follower_as = self.driver.find_elements_by_class_name('UserLink-link')  # 被关注的人主页链接元素列表
            follower_as = [follower_as[i] for i in range(len(follower_as)) if i % 2 == 0]
        print(root_user, " follower number: ", len(followers))

        # 获取following follower的ids, 并添加到root_user的变量中
        followings_id = [u.id for u in followings]
        followers_id = [u.id for u in followers]
        root_user.following_ids = followings_id
        root_user.follower_ids = followers_id
        return followings, followers

    def search_selenium(self, root_user):
        # TODO 模拟登录
        self.login()
        # 访问当前用户页面
        type_ = root_user.type
        url_id = root_user.url_id
        following_url = 'https://www.zhihu.com/' + type_ + '/' + url_id + '/following'  # 关注的
        follower_url = 'https://www.zhihu.com/' + type_ + '/' + url_id + '/followers'  # 被关注的
        self.url_ids.add(url_id)
        self.driver.get(following_url)
        try:
            # 获取当前用户信息
            agreed_divs = self.driver.find_elements_by_class_name('css-vurnku')
            agreed_num = 0
            for agreed_div in agreed_divs:
                if agreed_div.text[:2] == "获得":
                    agreed_num = int(agreed_div.text.split(' ')[1].replace(',', ''))
                    break
            name_span = self.driver.find_element_by_class_name('ProfileHeader-name')
            name = name_span.text
            root_user.agreed_num = agreed_num  # 获取点赞数
            root_user.name = name  # 获取名字
            print(root_user)
            # 获取关注列表和被关注列表的信息并将id列表加入root_user
            followings, followers = self.get_follow(root_user, following_url, follower_url)
            # 保存当前User到json,并保存到文件
            self.save(root_user, file_name="./data/zhihu-user.json")
            # 搜索关注和被关注列表的用户（深度优先搜索）
            for u in followings + followers:
                if u.url_id not in self.url_ids:
                    self.search_selenium(u)
        except NoSuchElementException as e:
            print(e)
        finally:
            self.driver.close()


def main():
    z = ZhihuUsership()
    z.search_selenium(User(url_id='jiaze-li', _type='people'))


if __name__ == '__main__':
    main()
```

