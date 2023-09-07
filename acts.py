import aiohttp
import asyncio
from lxml import html
import undetected_chromedriver as uc
import json
import os
from tqdm import tqdm
import hashlib

CONCURRENCY_LIMIT = 10

data = {
    'Acts': 'https://sso.agc.gov.sg/Browse/Act/Current/All?SortBy=Title&SortOrder=ASC',
    # 'Subsidiary Legislation': 'https://sso.agc.gov.sg/Browse/SL/Current/All?PageSize=20',
    # 'Acts Supplement': 'https://sso.agc.gov.sg/Browse/Acts-Supp/Published/All?PageSize=20',
    # 'Bills Supplement': 'https://sso.agc.gov.sg/Browse/Bills-Supp/Published/All?PageSize=20',
    # 'Subsidiary Legislation Supplement': 'https://sso.agc.gov.sg/Browse/SL-Supp/Published/All?PageSize=20',
    # 'Revised Editions of Acts': 'https://sso.agc.gov.sg/Browse/Act-Rev/Published/All?PageSize=20',
    # 'Revised Editions of Subsidiary Legislation': 'https://sso.agc.gov.sg/Browse/SL-Rev/Published/All?PageSize=20',

}


async def create_driver() -> uc.Chrome:
    '''Function to create webdriver for rendering webpages

    Returns:
    driver (uc.Chrome): Webdriver to render the page before scrapping
    '''
    options = uc.ChromeOptions()
    options.add_argument('--incognito')
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")

    # added new options since issues in Jan 2023 #743
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-application-cache")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--headless=new") #can enable or disable headless, up to your needs

    return uc.Chrome(options=options, version_main=115)


def calculate_md5_hash(data):
    md5_hash = hashlib.md5()
    md5_hash.update(data.encode())
    return md5_hash.hexdigest()


async def get_info_from_page(item: dict, driver: uc.Chrome, session: aiohttp.ClientSession):
    '''Using the webdriver get to the page, collect the data
    Parameters:
    item (dict): Dictionary containing the title and the url
    driver (uc.Chrome): Webdriver to render the page before scrapping

    Returns:
    item (dict): Modified dictionary with the data from page
    driver (uc.Chrome): Webdriver to render the page before scrapping
    '''
    driver.get(f'{item["url"]}?WholeDoc=1')
    await asyncio.sleep(10)
    tree = html.fromstring(driver.page_source)
    main = tree.xpath('//div[@id="legisContent"]')

    if main:
        front = main[0].xpath('./div/div[@class="front"]')
        if front:
            front_string = front[0].xpath('.//text()')
        else:
            front_string = ''
        item['front'] = front_string
        body = main[0].xpath('./div/div[@class="body"]')
        if body:
            tables_in_body = body[0].xpath('./div')
            body_data = []
            for i in range(len(tables_in_body)):
                item_table = {}
                tbls = tables_in_body[i].xpath('./table')  # gather text
                if len(tbls) == 2:
                    try:
                        key = tbls[0].xpath('./tbody/tr/td//text()')[0]
                        value = tbls[1].xpath('./tbody/tr/td//text()')
                        item_table[key] = value
                    except IndexError:
                        print('Empty key, saving in raw as well')
                else:
                    try:
                        value = tbls[0].xpath('./tbody/tr/td//text()')
                        body_data[i - 1][key] = value
                    except IndexError:
                        print('Table is broken, saving as raw as well', item["url"])
                    except UnboundLocalError:
                        print(
                            'No table in document, will be saved as raw as well')
                body_data.append(item_table)
                item['body'] = body_data
            tables_in_body = main[0].xpath(
                './div/div[@class="body"]//text()')
            item['body_raw'] = tables_in_body
        else:
            print(f'No body found {driver.current_url}')
        tail = main[0].xpath('./div/div[@class="tail"]//text()')
        if tail:
            item['tail_data'] = tail
    else:
        print(f'No main body {driver.current_url}')

    elms = tree.xpath('//div[@class="prov1"]')

    texts = []
    for x in elms:
        section_number = x.xpath(".//strong")[0].text_content()
        texts.append({"section": x.text_content(), "section_number": section_number})

    item["texts"] = texts
    return item, driver


async def get_number_of_pages(driver: uc.Chrome, session: aiohttp.ClientSession) -> int:
    tree = html.fromstring(driver.page_source)
    number = tree.xpath(
        '//div[@class="col-sm-12 no-side-padding page-count"]/text()')

    if number:
        return int(number[0].strip().split(' ')[3])
    else:
        print(
            f'No number of pages was found for {driver.current_url}\nReturning 1')
        return 1


async def get_urls_from_single_page(driver: uc.Chrome, session: aiohttp.ClientSession) -> list[dict]:
    tree = html.fromstring(driver.page_source)
    urls = tree.xpath(
        '//table[@class="table browse-list"]/tbody/tr/td[not(@class)]/a[@class="non-ajax"]/@href')
    tags = tree.xpath(
        '//table[@class="table browse-list"]/tbody/tr/td[not(@class)]/a[@class="non-ajax"]/text()')
    inter_data = []
    for i in range(len(urls)):
        item = {}
        item['url'] = f'https://sso.agc.gov.sg{urls[i]}'
        item['title'] = tags[i]
        inter_data.append(item)
    print(f'Page: {driver.current_url}, urls scraped: {len(inter_data)}')
    return inter_data


async def main():
    async with aiohttp.ClientSession() as session:
        driver = await create_driver()

        for key, val in data.items():
            print(f'Now scraping {key}\nStarting url:{val}')

            directory_name = val.split("/")[-1]

            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            driver.get(val)

            final = []
            for i in range(await get_number_of_pages(driver=driver, session=session)):
                driver.get(val.replace('/All', f'/All/{i}?'))
                await asyncio.sleep(10)
                final.extend(await get_urls_from_single_page(driver=driver, session=session))

            async with asyncio.Semaphore(CONCURRENCY_LIMIT):
                tasks = [get_info_from_page(item, driver=driver, session=session) for item in final]

                for item, _ in tqdm(await asyncio.gather(*tasks)):
                    url = item["url"].replace("https://", "").replace("/", " ")
                    file_path = f"{directory_name}/{url}.json"

                    if os.path.exists(file_path):
                        print(f"Skipped {file_path}!")
                        continue

                    item["Name of data"] = item["title"]
                    data_json = json.dumps(item)
                    md5_hash = calculate_md5_hash(data_json)
                    item["md5 hash"] = md5_hash

                    with open(file_path, "w") as f:
                        json.dump(item, f)

                with open(f'{directory_name}.json', 'w') as f:
                    json.dump(final, f)
                print(f'Data for {key} was collected, saved in {directory_name}.json\n')

if __name__ == '__main__':
    asyncio.run(main())

# https://sso.agc.gov.sg/Act/IRDA2018?WholeDoc=1#xy-