from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Set up the Selenium WebDriver
# driver = webdriver.Chrome()  # You may need to adjust this based on your browser and WebDriver

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')  # Add other options as needed
driver = webdriver.Chrome(options=chrome_options)

# Navigate to the webpage
url = "https://github.com/SimplifyJobs/Summer2024-Internships"
driver.get(url)

# Wait for the article element to be present in the DOM
try:
    article = WebDriverWait(driver, 3).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'markdown-body'))
    )

    # Extract the HTML content of the article
    article_html = article.get_attribute("outerHTML")

    # Use BeautifulSoup to parse the HTML of the article
    soup = BeautifulSoup(article_html, 'html.parser')

    # Find the table tag within the article
    table_tag = soup.find('table')

    if table_tag:
        # Print the table tag
        # print(table_tag)
        with open("table.html", "w") as f:
            f.write(str(table_tag))
    else:
        print("Table not found within the article.")

finally:
    # Close the browser
    driver.quit()
