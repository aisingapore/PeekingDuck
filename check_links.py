import os
import urllib.request
from bs4 import BeautifulSoup
import markdown
import re


def get_html():
    lst_html = []
    # search path is hard coded with ref of this script locaiton
    for root, dirs, files in os.walk(os.path.join(".","docs","build","html"), topdown=False):
        for name in files:
            if name[-4:] == "html":
                filepath = os.path.join(root, name)
                print(filepath)
                lst_html.append((filepath, root))
    return lst_html

def get_md():
    lst_md=[]
    # search path is hard coded with ref of this script locaiton
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name[-2:] == "md":
                filepath = os.path.join(root, name)
                print(filepath)
                lst_md.append((filepath, root))
    return lst_md

def check_htmls(lst_html):
    faulty_links =[]
    for filepath, root in lst_html:
    
        with open(filepath, "r", encoding='utf-8') as f:
            file = f.read()
            file_2_html = markdown.markdown(file)
            soup = BeautifulSoup( file_2_html,'html.parser')
            img_soup = soup.find_all("img")
            href_soup = soup.find_all("a")
            img_links = [tag["src"] for tag in img_soup if tag.has_attr("src")]
            href_links = [tag["href"] for tag in href_soup if tag.has_attr("href")]
            final = img_links+href_links
            final = [txt for txt in final if ("." in txt) and ("#" not in txt)] # remove leftover artifacts

        # for link in final:
        #     # if filepath is relative path to a html file or to _source/_static/_module folder
        #     # or to a local yml file such as run_config.yml
        #     if link.startswith(".") or link[-4:] == "html" \
        #         or link.startswith("_") or link[-3:] == "yml": 
        #        check = os.path.exists(os.path.join(root, link))

        #        if not check:
        #            faulty_links.append((filepath , link))

        #     else:
        #         try:
        #             resp = urllib.request.urlopen(link)
        #         except Exception as e:
        #             faulty_links.append((filepath , link, e.code))

        for link in final:
            # if link is a https link, run request.urlopen
            if link[:5] == "https":
                try:
                    resp = urllib.request.urlopen(link)
                except Exception as e:
                    # here i filter only 404 error
                    # if u want catch all then u remove if statement
                    if e.code == 404: 
                        # filepath is the current file being parsed
                        # link is the link found in the current parsed file
                        # e.code is the execption code such as 404,403...
                        faulty_links.append((filepath , link, e.code))
            
            else: 
                check = os.path.exists(os.path.join(root, link))

                if not check:
                    # filepath is the current file being parsed
                    # link is the link found in the current parsed file
                    # root is the root folder of the filepath of current file
                    faulty_links.append((filepath , link, root))

        print(f"complete {filepath}")
    
    return faulty_links

htmls = get_html()
mds = get_md()
faulty_links_htmls = check_htmls(htmls)
faulty_links_mds = check_htmls(mds)

faulty_link_final = faulty_links_htmls + faulty_links_mds

# print(faulty_links_htmls)
# print(faulty_links_mds)
print(faulty_link_final)
