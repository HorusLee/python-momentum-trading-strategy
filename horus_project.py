# -*- coding: utf-8 -*-

"""
@author: Zihao Li
Class: CS677 - Spring
Date: Sun Apr 26, 2020
Term Project Assignment
Description of Problem:
Momentum Trading Strategy
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from tqdm import tqdm

# initialize some factors
dataset = '1995_2020_major_1000.xlsx'
filename = 'all_return.csv'
sp_500 = 'sp-500-historical-annual-returns.csv'

formation_period = np.array(range(1, 12))
holding_period = 12
portfolio_size = [1, 5, 10, 50, 100, 500]
risk_free_rate = 0.01
trading_period = [1, 2, 3, 4, 6, 12]
year_span = np.array(range(1996, 2020))

all_return = np.array(
	[[[0.0] * len(trading_period)] * len(portfolio_size)] * len(year_span))
figure_size = (12, 24)
line_width = 1

historical_return = pd.read_csv(sp_500, index_col=0)
return_2020 = np.array([[0.0] * 2] * len(portfolio_size))
return_heaps = np.array([[0.0] * len(trading_period)] * len(portfolio_size))
sharp_ratios = np.array([[0.0] * len(trading_period)] * len(portfolio_size))
test_portfolio_size = np.array(range(1, 927))  # 1 ~ 927
trading_results = np.array(
	[[[0.0] * (len(year_span) + 1)] * len(trading_period)] * len(portfolio_size))


def get_performance(formation_data_1, formation_data_2):
	"""
	Loop for all tickers to get sorted performance.
	"""
	performance = []
	dictionary = {}
	for i in range(12):
		performance.append(copy.deepcopy(dictionary))
	year_performance = []

	# loop for first formation year
	for index1, column1 in formation_data_1.items():

		# loop for second formation year
		for index2, column2 in formation_data_2.items():
			if index1 == index2:

				# get the price list
				column = []
				for i in range(12):
					column.append(column1[i + 1])
				for i in range(len(column2) - 1):
					column.append(column2[i + 1])

				# get the performance
				for i in range(len(column) - 12):
					month_return = []
					for f in formation_period:
						month_return.append((column[i + f] - column[i + f - 1])
						                    / column[i + f - 1] + 1)
					performance[i][index1] = gmean(np.array(month_return)) - 1

	# sort all the performances
	for i in range(12):
		year_performance.append(
			sorted(performance[i].items(), key=lambda x: x[1], reverse=True))
	return year_performance


def get_return():
	"""
	The function will get all returns and put them in a csv file.
	"""
	# loop for year
	for y in tqdm(range(len(year_span)), ncols=50):
		train_data = pd.read_excel(dataset, sheet_name=str(year_span[y] - 1))
		test_data_1 = pd.read_excel(dataset, sheet_name=str(year_span[y]))
		test_data_2 = pd.read_excel(dataset, sheet_name=str(year_span[y] + 1))
		sorted_performance = get_performance(train_data, test_data_1)

		# loop for portfolio size
		for s in range(len(portfolio_size)):

			# loop for trading period
			for t in range(len(trading_period)):
				transactions = int(holding_period / trading_period[t])
				portfolio_return = np.array(
					[[0.0] * portfolio_size[s]] * transactions)

				# loop for transactions
				for n in range(transactions):

					# loop for best portfolios
					for p in range(0, portfolio_size[s]):
						ticker_name = sorted_performance[trading_period[t] * n][p][0]

						# loop for first trading year
						for index1, column1 in test_data_1.items():
							if index1 == ticker_name:

								# loop for second trading year
								for index2, column2 in test_data_2.items():
									if index2 == ticker_name:

										# get the price list
										column = []
										for i in range(12):
											column.append(column1[i + 1])
										column.append(column2[1])

										# compute portfolio return
										portfolio_return[n][p] = \
											(column[trading_period[t] * (n + 1)] -
											 column[trading_period[t] * n]) / \
											column[trading_period[t] * n]

						# fill non values with 0
						if portfolio_return[n][p] >= -1:
							print(end='')
						else:
							portfolio_return[n][p] = 0.0

				# compute period returns
				period_return = []
				for i in range(len(portfolio_return)):
					period_return.append(gmean(portfolio_return[i] + 1) - 1)

				# compute all returns
				year_return = 1
				for i in range(len(period_return)):
					year_return *= period_return[i] + 1
				all_return[y][s][t] = year_return - 1

	# save all returns as csv file
	all_return_reshape = pd.DataFrame(
		all_return.reshape(
			len(year_span) * len(portfolio_size), len(trading_period)))
	all_return_reshape.to_csv(filename)
	print("------------------------------Finish------------------------------")


def plot_return():
	"""
	The function will plot all returns by using to different ways.
	"""
	# load csv file
	returns = pd.read_csv(filename, index_col=0)

	# plot return vs. year for different portfolio size
	plt.figure(figsize=figure_size)
	plt.suptitle("return vs. year for trading period and portfolio size")
	for s in range(len(portfolio_size)):
		plt.subplot("{}{}{}".format(len(portfolio_size), 1, s + 1))
		for t in range(len(trading_period)):
			return_line = []
			for y in range(len(year_span)):
				return_line.append(returns[str(t)][y * len(portfolio_size) + s])
			plt.plot(year_span, return_line, lw=line_width,
			         label='trading period = {}'.format(trading_period[t]))
		plt.plot(year_span, historical_return[
		                    year_span[0] - 1928:year_span[-1] - 1927] / 100,
		         c='k', lw=line_width, label='historical annual returns')
		plt.grid(ls='--')
		plt.xticks(year_span)
		plt.ylabel("portfolio size = {}".format(portfolio_size[s]))
		plt.legend()
	plt.xlabel("year")
	plt.savefig("portfolio_size.png")
	plt.show()

	# plot return vs. year for different trading period
	plt.figure(figsize=figure_size)
	plt.suptitle("return vs. year for portfolio size and trading period")
	for t in range(len(trading_period)):
		plt.subplot("{}{}{}".format(len(trading_period), 1, t + 1))
		for s in range(len(portfolio_size)):
			return_line = []
			for y in range(len(year_span)):
				return_line.append(returns[str(t)][y * len(portfolio_size) + s])
			plt.plot(year_span, return_line, lw=line_width,
			         label='portfolio size = {}'.format(portfolio_size[s]))
		plt.plot(year_span, historical_return[
		                    year_span[0] - 1928:year_span[-1] - 1927] / 100,
		         c='k', lw=line_width, label='historical annual returns')
		plt.grid(ls='--')
		plt.xticks(year_span)
		plt.ylabel("trading period = {}".format(trading_period[t]))
		plt.legend()
	plt.xlabel("year")
	plt.savefig("trading_period.png")
	plt.show()


def return_heap():
	"""
	The function will plot return heaps for different combinations.
	"""
	# load csv file
	returns = pd.read_csv(filename, index_col=0)

	# get reshaped returns
	returns_reshape = np.array(pd.DataFrame(np.array(returns).reshape(
		len(year_span), len(trading_period) * len(portfolio_size))).T)

	# get return heaps
	for i in range(len(returns_reshape)):
		return_heaps[int(i / len(trading_period))][i % len(trading_period)] = \
			np.prod(returns_reshape[i] + 1) - 1

	# plot return heaps for different trading period and portfolio size
	plt.title("Return Heaps for Different Combinations")
	for s in range(len(portfolio_size)):
		for t in range(len(trading_period)):
			plt.scatter(t, s, c='g' if return_heaps[s][t] > 0 else 'r',
			            s=abs(return_heaps[s][t]) * 50)
	plt.xticks(range(len(trading_period)), trading_period)
	plt.yticks(range(len(portfolio_size)), portfolio_size)
	plt.xlabel("trading period")
	plt.ylabel("portfolio size")
	plt.show()


def sharp_ratio():
	"""
	The function will plot all sharp ratios for different combinations.
	"""
	# load csv file
	returns = pd.read_csv(filename, index_col=0)
	# historical_return = pd.read_csv(sp_500, index_col=0)

	# get reshaped returns
	returns_reshape = np.array(pd.DataFrame(np.array(returns).reshape(
		len(year_span), len(trading_period) * len(portfolio_size))).T)

	# get sharp ratios
	for i in range(len(returns_reshape)):
		sharp_ratios[int(i / len(trading_period))][i % len(trading_period)] = \
			(gmean(returns_reshape[i] + 1) - 1 - risk_free_rate) / \
			np.std(returns_reshape[i])

	# plot sharp ratios for different trading period and portfolio size
	plt.title("Sharp Ratios for Different Combinations")
	for s in range(len(portfolio_size)):
		for t in range(len(trading_period)):
			plt.scatter(t, s, c='g' if sharp_ratios[s][t] > 0 else 'r',
			            s=abs(sharp_ratios[s][t]) * 1000)
			plt.text(t, s, sharp_ratios[s][t].round(2), c='b')
	plt.xticks(range(len(trading_period)), trading_period)
	plt.yticks(range(len(portfolio_size)), portfolio_size)
	plt.xlabel("trading period")
	plt.ylabel("portfolio size")
	plt.savefig("sharp_ratio.png")
	plt.show()


def test_2020():
	"""
	The function will plot portfolio size vs. average return for year 2020.
	"""
	train_data = pd.read_excel(dataset, sheet_name='2019')
	test_data = pd.read_excel(dataset, sheet_name='2020')
	portfolio_return = []
	test_return = []

	# get the performance
	performance = {}
	for index, column in train_data.items():
		month_return = []
		for f in formation_period:
			month_return.append((column[f + 1] - column[f]) / column[f] + 1)
		performance[index] = gmean(np.array(month_return)) - 1
	sorted_performance = sorted(
		performance.items(), key=lambda x: x[1], reverse=True)

	# loop for portfolio
	for s in range(len(sorted_performance)):
		ticker_name = sorted_performance[s][0]
		for index, column in test_data.items():
			if index == ticker_name:
				portfolio_return.append((column[3] - column[1]) / column[1] + 1)

	# loop for portfolio return to get test return
	for r in range(1, len(portfolio_return) + 1):
		test_return.append(gmean(np.array(portfolio_return[:r])) - 1)

	# plot the results
	for i in test_portfolio_size:
		plt.bar(i, test_return[i - 1])
	plt.title("Portfolio Size VS. Average Return for Year 2020")
	plt.xlabel("portfolio size")
	plt.ylabel("average return")
	plt.show()


def trading_result():
	"""
	The function will plot trading results (cumulative return) of all combinations.
	"""
	# load csv file
	returns = pd.read_csv(filename, index_col=0)
	new_year_span = np.array(range(1996, 2021))
	beach_mark = []

	# get reshaped returns
	returns_reshape = np.array(pd.DataFrame(np.array(returns).reshape(
		len(year_span), len(trading_period) * len(portfolio_size))).T)

	# get cumulative returns
	for i in range(len(returns_reshape)):
		for j in range(len(returns_reshape[i]) + 1):
			trading_results[int(i / len(trading_period))][
				i % len(trading_period)][j] = \
				np.prod(returns_reshape[i][:j] + 1)
	for i in range(len(new_year_span)):
		beach_mark.append(np.prod(
			historical_return[year_span[0] - 1928:year_span[0] - 1928 + i] / 100 + 1))

	# plot cumulative returns
	for j in range(len(trading_results[4])):
		plt.plot(new_year_span, trading_results[4][j],
		         label="portfolio size = {}, trading period = {}".
		         format(portfolio_size[4], trading_period[j]))
		plt.text(new_year_span[-1], trading_results[4][j][-1],
		         trading_results[4][j][-1].round(1))
	plt.plot(new_year_span, beach_mark, c='k',
	         label="bench mark - sp 500 cumulative returns")
	plt.text(new_year_span[-1], beach_mark[-1], float(beach_mark[-1].round(1)))
	plt.grid(ls='--')
	plt.xticks(new_year_span, rotation=90)
	plt.yticks(range(0, 11))
	plt.xlabel('year')
	plt.ylabel("cumulative return")
	plt.title("Cumulative Returns for Portfolio Size = 100")
	plt.legend()
	plt.savefig("trading_result.png")
	plt.show()


if __name__ == '__main__':
	# get_return()
	# plot_return()
	# return_heap()
	# sharp_ratio()
	# test_2020()
	trading_result()
