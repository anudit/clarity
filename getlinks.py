from pprint import pprint
from urllib.parse import urldefrag, urlparse

import undetected_chromedriver as uc
from bs4 import BeautifulSoup as bs
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

ua = UserAgent()

def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = uc.Chrome(options=options)
    driver.implicitly_wait(2)
    return driver

def get_from_sel(url):

    try:
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        
        src = driver.page_source
        return src
    except:
        return False

def get_links_on_page(SEARCH_URL, ROOT):

    req = get_from_sel(SEARCH_URL)
    soup = bs(req, 'html.parser') 
    
    finds = soup.find_all('a', href=True)
    
    hrefs = []
    for link in finds :
        href = link['href']
        cleaned_href = urldefrag(href if 'http' in href else 'https://'+urlparse(SEARCH_URL).hostname + href).url
        if "mailto" in cleaned_href or "https://" not in cleaned_href:
            continue
        else:
            hrefs.append(cleaned_href)
    
    hrefs = list(set([x for x in hrefs if ROOT in x]))

    return hrefs

def get_links_on_pages(SEARCH_URLS = [], ROOT=""):
    links = [];
    for link in tqdm(SEARCH_URLS):
        links += get_links_on_page(link, ROOT)
    
    return list(set(links))

def depth_lookup(start, n):

    ROOT = urlparse(start).hostname;
    global driver;
    driver = create_driver();

    links = [start]
    for x in range(n):
        links = get_links_on_pages(links, ROOT)
        print('Found', len(links), 'on Level', x+1)
    
    driver.quit()

    return links


# # SEARCH_URL = 'https://www.electronicid.eu/en/blog'
# SEARCH_URL = 'https://0xparc.org/blog';
# ROOT = urlparse(SEARCH_URL).hostname;

# links = depth_lookup(SEARCH_URL, 3)
# pprint(links)

