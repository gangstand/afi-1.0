import requests
from bs4 import BeautifulSoup



# Основной класс
class Currency:
	DOLLAR_RUB = 'https://www.google.com/search?sxsrf=ALeKk01NWm6viYijAo3HXYOEQUyDEDtFEw%3A1584716087546&source=hp&ei=N9l0XtDXHs716QTcuaXoAg&q=%D0%B4%D0%BE%D0%BB%D0%BB%D0%B0%D1%80+%D0%BA+%D1%80%D1%83%D0%B1%D0%BB%D1%8E&oq=%D0%B4%D0%BE%D0%BB%D0%BB%D0%B0%D1%80+&gs_l=psy-ab.3.0.35i39i70i258j0i131l4j0j0i131l4.3044.4178..5294...1.0..0.83.544.7......0....1..gws-wiz.......35i39.5QL6Ev1Kfk4'
	headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}

	current_converted_price = 0

	def __init__(self):
		self.current_converted_price = float(self.get_currency_price().replace(",", "."))


	def get_currency_price(self):
		full_page = requests.get(self.DOLLAR_RUB, headers=self.headers)
		soup = BeautifulSoup(full_page.content, 'html.parser')
		convert = soup.findAll("span", {"class": "DFlfde", "class": "SwHCTb", "data-precision": 2})
		return convert[0].text

	def check_currency(self):
		currency = float(self.get_currency_price().replace(",", "."))

		print("Курс: доллар = " + str(currency))

currency = Currency()
currency.check_currency()

# Основной класс
class Currency1:
	EURO_RUB = 'https://www.google.com/search?q=курс+евро&sxsrf=AOaemvIHHUkd7Q0URG2r9L94243sihMFzQ%3A1636790156011&source=hp&ei=i2-PYfqAO4iprgSpgJqIDA&iflsig=ALs-wAMAAAAAYY99nHQg3MZXwCxzP1F2CPnoEGNmOFQx&oq=курс+евро&gs_lcp=Cgdnd3Mtd2l6EAMyDQgAEIAEELEDEEYQggIyCwgAEIAEELEDEIMBMggIABCABBCxAzIICAAQgAQQsQMyCwgAEIAEELEDEIMBMgUIABCABDIFCAAQsQMyCwgAEIAEELEDEMkDMggIABCABBCxAzIICAAQgAQQsQM6CAguELEDEIMBOg4ILhCABBCxAxDHARDRAzoICAAQsQMQgwE6EAgAEIAEELEDEIMBEEYQggJQAFiJG2DmH2gAcAB4AIABvwGIAZ0HkgEDNy4ymAEAoAEB&sclient=gws-wiz&ved=0ahUKEwi68sW-7pT0AhWIlIsKHSmABsEQ4dUDCAY&uact=5'
	headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}

	current_converted_price = 0

	def __init__(self):
		self.current_converted_price = float(self.get_currency_price().replace(",", "."))


	def get_currency_price(self):
		full_page = requests.get(self.EURO_RUB, headers=self.headers)
		soup = BeautifulSoup(full_page.content, 'html.parser')
		convert = soup.findAll("span", {"class": "DFlfde", "class": "SwHCTb", "data-precision": 2})
		return convert[0].text

	def check_currency(self):
		currency = float(self.get_currency_price().replace(",", "."))

		print("Курс: евро = " + str(currency))

currency = Currency1()
currency.check_currency()