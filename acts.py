import asyncio
import glob
import json
import os

import aiohttp
import undetected_chromedriver as uc
from lxml import html
from tqdm import tqdm

data = {
    "Acts": "https://sso.agc.gov.sg/Browse/Act/Current/All?SortBy=Title&SortOrder=ASC",
    "Subsidiary Legislation": "https://sso.agc.gov.sg/Browse/SL/Current/All?PageSize=20",
    "Acts Supplement": "https://sso.agc.gov.sg/Browse/Acts-Supp/Published/All?PageSize=20",
    "Bills Supplement": "https://sso.agc.gov.sg/Browse/Bills-Supp/Published/All?PageSize=20",
    "Subsidiary Legislation Supplement": "https://sso.agc.gov.sg/Browse/SL-Supp/Published/All?PageSize=20",
    "Revised Editions of Acts": "https://sso.agc.gov.sg/Browse/Act-Rev/Published/All?PageSize=20",
    "Revised Editions of Subsidiary Legislation": "https://sso.agc.gov.sg/Browse/SL-Rev/Published/All?PageSize=20",
}


async def create_driver() -> uc.Chrome:
    """Function to create webdriver for rendering webpages
    Returns:
    driver (uc.Chrome): Webdriver to render the page before scrapping
    """
    uc.find_chrome_executable()
    options = uc.ChromeOptions()
    options.add_argument("--incognito")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")

    # added new options since issues in Jan 2023 #743
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-application-cache")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--headless=new")
    options.binary_location = "/usr/bin/chromium-browser"
    print("Chrome Binary Location:", options.binary_location)
    return uc.Chrome(options=options, version_main=117)


async def get_info_from_page(item: dict, driver: uc.Chrome, session: aiohttp.ClientSession):
    """Using the webdriver get to the page, collect the data
    Parameters:
    item (dict): Dictionary containing the title and the url
    driver (uc.Chrome): Webdriver to render the page before scrapping
    Returns:
    item (dict): Modified dictionary with the data from page
    driver (uc.Chrome): Webdriver to render the page before scrapping
    """
    driver.get(f'{item["url"]}?WholeDoc=1')
    await asyncio.sleep(30)
    tree = html.fromstring(driver.page_source)
    main = tree.xpath('//div[@id="legisContent"]')

    if main:
        front = main[0].xpath('./div/div[@class="front"]')
        if front:
            front_string = front[0].xpath(".//text()")
        else:
            front_string = ""
        item["front"] = front_string
        body = main[0].xpath('./div/div[@class="body"]')
        if body:
            tables_in_body = body[0].xpath("./div")
            body_data = []
            for i in range(len(tables_in_body)):
                item_table = {}
                tbls = tables_in_body[i].xpath("./table")  # gather text
                if len(tbls) == 2:
                    try:
                        key = tbls[0].xpath("./tbody/tr/td//text()")[0]
                        value = tbls[1].xpath("./tbody/tr/td//text()")
                        item_table[key] = value
                    except IndexError:
                        print("Empty key, saving in raw as well")
                else:
                    try:
                        value = tbls[0].xpath("./tbody/tr/td//text()")
                        body_data[i - 1][key] = value
                    except IndexError:
                        print("Table is broken, saving as raw as well", item["url"])
                    except UnboundLocalError:
                        print("No table in document, will be saved as raw as well")
                body_data.append(item_table)
                item["body"] = body_data
            tables_in_body = main[0].xpath('./div/div[@class="body"]//text()')
            item["body_raw"] = tables_in_body
        else:
            print(f"No body found {driver.current_url}")
        tail = main[0].xpath('./div/div[@class="tail"]//text()')
        if tail:
            item["tail_data"] = tail
    else:
        print(f"No main body {driver.current_url}")

    elms = tree.xpath('//div[@class="prov1"]')

    texts = []
    for x in elms:
        section_number = x.xpath(".//strong")[0].text_content()
        texts.append({"section": x.text_content(), "section_number": section_number})

    item["texts"] = texts
    return item, driver


async def get_number_of_pages(driver: uc.Chrome, session: aiohttp.ClientSession) -> int:
    tree = html.fromstring(driver.page_source)
    number = tree.xpath('//div[@class="col-sm-12 no-side-padding page-count"]/text()')

    if number:
        return int(number[0].strip().split(" ")[3])
    else:
        print(f"No number of pages was found for {driver.current_url}\nReturning 1")
        return 1


async def get_urls_from_single_page(driver: uc.Chrome, session: aiohttp.ClientSession) -> list[dict]:
    tree = html.fromstring(driver.page_source)
    urls = tree.xpath('//table[@class="table browse-list"]/tbody/tr/td[not(@class)]/a[@class="non-ajax"]/@href')
    tags = tree.xpath('//table[@class="table browse-list"]/tbody/tr/td[not(@class)]/a[@class="non-ajax"]/text()')
    inter_data = []
    for i in range(len(urls)):
        item = {}
        item["url"] = f"https://sso.agc.gov.sg{urls[i]}"
        item["title"] = tags[i]
        inter_data.append(item)
    print(f"Page: {driver.current_url}, urls scraped: {len(inter_data)}")
    return inter_data


def combined(path, name):
    file_paths = glob.glob(f"{path}/*.json")
    all_json_data = []

    for file_path in file_paths:
        with open(file_path, "r") as file:
            json_data = json.load(file)
            all_json_data.append(json_data)

    combined_directory = "data/combined"
    if not os.path.exists(combined_directory):
        os.makedirs(combined_directory)

    output_file = f"{combined_directory}/combined{name}.json"
    with open(output_file, "w") as outfile:
        json.dump(all_json_data, outfile, indent=2)


async def scrape_and_save_data(key, val):
    async with aiohttp.ClientSession() as session:
        driver = await create_driver()

        print(f"Now scrapping {key}\nStarting url:{val}")

        directory_name = "data/" + key

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        driver.get(val)

        final = []
        for i in range(await get_number_of_pages(driver=driver, session=session)):
            driver.get(val.replace("/All", f"/All/{i}?"))
            await asyncio.sleep(10)
            final.extend(await get_urls_from_single_page(driver=driver, session=session))

        for j in tqdm(range(len(final))):
            url = final[j]["url"].replace("https://", "").replace("/", " ")
            if os.path.exists(f"{key}/{url}.json"):
                print(f"Skipped!")
                continue

            final[j], driver = await get_info_from_page(final[j], driver=driver, session=session)

            with open(f"{directory_name}/{url}.json", "w") as f:
                json.dump(final[j], f)

        with open(f"data/{key}.json", "w") as f:
            json.dump(final, f)

        print(f"Data for {key} was collected, saved in {directory_name}.json\n")

        combined(directory_name, key)


async def main():
    tasks = []
    for key, val in data.items():
        task = scrape_and_save_data(key, val)
        tasks.append(task)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
# https://sso.agc.gov.sg/Act/IRDA2018?WholeDoc=1#xy-
