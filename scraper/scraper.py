from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json

BASE_URL = "https://shop.kingarthurbaking.com"
CATEGORY_URL = f"{BASE_URL}/mixes"


def get_product_urls(page):
    urls = set()

    links = page.locator("a[href*='/items/']")

    for i in range(links.count()):
        href = links.nth(i).get_attribute("href")

        if not href:
            continue

        if href.startswith("/"):
            href = BASE_URL + href

        urls.add(href)

    return urls


def load_all_products(page):
    """
    Keep scrolling until no new product URLs
    appear for several rounds.
    """

    all_urls = set()

    no_new_rounds = 0
    max_no_new_rounds = 8

    while True:

        page.mouse.wheel(0, 8000)

        page.wait_for_timeout(3000)

        current_urls = get_product_urls(page)

        old_count = len(all_urls)

        all_urls.update(current_urls)

        new_count = len(all_urls)

        print(
            f"Products collected: {new_count}"
        )

        if new_count == old_count:
            no_new_rounds += 1
        else:
            no_new_rounds = 0

        if no_new_rounds >= max_no_new_rounds:
            print(
                "No new products found. "
                "Finished scrolling."
            )
            break

    return list(all_urls)


def scrape_product(page, url):

    page.goto(
        url,
        wait_until="networkidle",
        timeout=60000
    )

    page.wait_for_timeout(2000)

    soup = BeautifulSoup(
        page.content(),
        "html.parser"
    )
    
    id = page.locator(".product-sku").first.inner_text(timeout=3000)

    h1 = soup.find("h1")

    title = (
        h1.get_text(strip=True)
        if h1
        else "N/A"
    )

    description = "N/A"

    try:
        description = page.locator(
            ".tab-content"
        ).first.inner_text(timeout=5000)
    except:
        pass
    
    rating = "N/A"
    
    try:
        rating = page.locator(
            ".rating"
        ).first.inner_text(timeout=5000)[:1]
    except:
        pass

    price = "N/A"

    selectors = [
        ".orig-price",
        ".price",
        ".price--withoutTax"
    ]

    for selector in selectors:

        try:

            if page.locator(selector).count():

                price = page.locator(
                    selector
                ).first.inner_text()

                break

        except:
            pass

    ingredients = "N/A"

    ingredient_selectors = [
        ".ingredients-html",
        "[data-test='ingredients']"
    ]

    for selector in ingredient_selectors:

        try:

            if page.locator(selector).count():

                ingredients = page.locator(
                    selector
                ).first.inner_text()
                
                ingredients = ingredients[13:]
                
                break

        except:
            pass
    if "Contains" in ingredients:
        i = ingredients.index("Contains")
        contains = ingredients[i+8:]
    else:
        contains = ""
    
    if "Yeast" in ingredients:
        i = ingredients.index("Yeast")
        yeast = ingredients[i+5:]
    else:
        yeast = ""

    if "Details" in description:
        i = description.index("Details")
        details = description[i+8:]
        description = description[:i]
    else:
        details = ""
        
    nutrition_link = "N/A"
    
    try:
        nutrition_link = soup.find('a', class_='nutrition-link')
        nutrition_link = nutrition_link.get('href')
    except:
        pass
    review = "N/A"
    
    try:
        review = soup.find('span', class_='reviews__label')
        review = review.get_text(strip=True)
        print(reveiw)
    except:
        pass
    image_list = []
    try:
        images = soup.find_all('img', class_='lazyautosizes ls-is-cached lazyloaded')
        for image in images:
            image_list.append(image.get("src"))
        
        print(image_list);
    except:
        pass
    
    return {
        "id": id,
        "name": title,
        "price": price,
        "description": description,
        "details": details,
        "ingredients": ingredients,
        "contains": contains,
        "rating": rating,
        "nutrition_link": nutrition_link,
        "review": review,
        "image_list": image_list,
        "url": url
    }


def main():

    all_products = []

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False
        )

        page = browser.new_page(
            viewport={
                "width": 1600,
                "height": 1200
            }
        )

        print("Opening category page...")

        page.goto(
            CATEGORY_URL,
            wait_until="domcontentloaded",
            timeout=60000
        )

        page.wait_for_timeout(5000)

        print("Loading all products...")

        product_urls = load_all_products(page)

        print(
            f"Total unique products: "
            f"{len(product_urls)}"
        )

        detail_page = browser.new_page()

        for index, url in enumerate(
            product_urls,
            start=1
        ):

            try:

                data = scrape_product(
                    detail_page,
                    url
                )

                all_products.append(data)

                print(
                    f"[{index}/{len(product_urls)}] "
                    f"{data['name']}"
                )

            except Exception as e:

                print(
                    f"Failed: {url}"
                )

                print(e)

        detail_page.close()

        with open(
            "./data/detailInfo.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                all_products,
                f,
                indent=2,
                ensure_ascii=False
            )

        browser.close()

        print(
            f"Saved "
            f"{len(all_products)} products"
        )


if __name__ == "__main__":
    main()