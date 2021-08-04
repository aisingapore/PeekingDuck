import os
import urllib.request
from bs4 import BeautifulSoup
import markdown
import re


def get_html():
    lst_html = []
    for root, dirs, files in os.walk(os.path.join(".","docs","build","html"), topdown=False):
        for name in files:
            if name[-4:] == "html":
                filepath = os.path.join(root, name)
                print(filepath)
                lst_html.append((filepath, root))
    return lst_html


test_root = os.path.join(".","docs","build","html","models")
test_file = os.path.join(".","docs","build","html","models","yolo.html")

def extract(text):

    pattern = r'src="(.*)"\s'
    #print(re.findall(pattern, '<img src="images/readme/zone_counting.gif" width="100%"/>'))
    result = re.findall(pattern , text)

    if len(result)!=0:
        return result[0]

def check_htmls(lst_html):
    
    for filepath, root in lst_html:
    
        with open(filepath, "r", encoding='utf-8') as f:
            file = f.read()
            file_2_html = markdown.markdown(file)
            soup = BeautifulSoup( file_2_html,'html.parser')
            img_soup = soup.find_all("img")
            href_soup = soup.find_all("a")
            img_links = [tag["src"] for tag in img_soup]
            href_links = [tag["href"] for tag in href_soup]
            final = img_links+href_links
            final = [txt for txt in final if "." in txt] # remove leftover artifacts

# print(href_soup)
# print(img_soup)
# #print(href_links)

# print(final)
# #print([txt for txt in ["a" , "b.c"] if "." in txt])



try:# req=urllib.request.Request(url="https://aisingapore.org/home/contact/")
    resp = urllib.request.urlopen('https://aisingapore.org/home/contact/')
    print(resp.status)
except Exception as e:
    print(f"{e}")
    print(f"{e.code}")

print(test_file)
print(test_root)
print(os.path.exists(os.path.join(test_root, '../index.html')))