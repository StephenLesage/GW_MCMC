"""
MCMC for fake GW data
"""

import os
import sys
import warnings
import numpy as np
from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['figure.figsize'] = (8,8)
warnings.filterwarnings("ignore")


MCMC_STEPS = 100000
Sn_range = (0,30)
i_range = (-5,5)


def pBar(iterations, prefix='', bar_length=40):
	"""
	example: for i in pBar(range(50), 'Computing:'):
	"""
	def show(current):
		remaining = int(bar_length*current/len(iterations))
		sys.stdout.write( '    %s [%s%s] %i/%i [%i%%]\r'%\
			(prefix, u'█'*remaining, '.'*(bar_length-remaining), current, len(iterations), (current/len(iterations))*100))
		sys.stdout.flush()  
	show(0)
	for index, value in enumerate(iterations):
		yield value
		show(index+1)
	sys.stdout.write('\n')
	sys.stdout.flush()
	return


def PSD(i, time_obs, S_alpha, alpha, S_beta, beta):

	fmin = 0.001
	fref = 0.25

	f = i/time_obs

	if f < fmin:
		f = fmin

	return np.sqrt(time_obs) * (S_alpha * (f/fref)**alpha + S_beta * (f/fref)**beta) * 2


def log_likelihood(data, theta, dataset_range):

	network_logL = 0.0
	for i in range(dataset_range[0], dataset_range[1]+1):
		network_logL += _log_likelihood(data[i], theta)

	return network_logL


def _log_likelihood(data, theta):

	Sn_alpha, alpha, Sn_beta, beta = theta

	chi2 = 0.0
	norm = 0.0
	Sn = np.zeros(len(data['dFFT']))

	for i in range(len(data['dFFT'])):
		Sn[i] = PSD(i, int(len(data['time'])), Sn_alpha, alpha, Sn_beta, beta)
		chi2 += np.real(data['dFFT'][i] * np.conjugate(data['dFFT'][i]) / Sn[i])
		norm -= 0.5 * np.log(Sn[i])

	return -0.5 * chi2 + norm


def log_prior(theta):

	Sn_alpha, alpha, Sn_beta, beta = theta

	if Sn_range[0] <= Sn_alpha <= Sn_range[1] and 
	   Sn_range[0] <= Sn_beta <= Sn_range[1] and 
	   i_range[0] <= alpha <= i_range[1] and 
	   i_range[0] <= beta <= i_range[1]:
		return 0.0
	else:
		return -np.inf


def gaussian_proposal(theta):

	Sn_alpha, alpha, Sn_beta, beta = theta

	Sn_alpha += np.random.normal(0,10)
	alpha += np.random.normal(0,0.5)
	Sn_beta += np.random.normal(0,10)
	beta += np.random.normal(0,0.5)

	return np.array([Sn_alpha, alpha, Sn_beta, beta])


def uniform_proposal():

	Sn_alpha = np.random.uniform(Sn_range[0], Sn_range[1])
	alpha = np.random.uniform(i_range[0], i_range[1])
	Sn_beta = np.random.uniform(Sn_range[0], Sn_range[1])
	beta = np.random.uniform(i_range[0], i_range[1])

	return np.array([Sn_alpha, alpha, Sn_beta, beta])


def pick_proposal(theta):

	if np.random.uniform(0,1) < 0.5:
		theta_next = uniform_proposal()
	else:
		theta_next = gaussian_proposal(theta)

	return theta_next


def mcmc(data, dataset_range, pBarLabel=''):

	## initialize chain
	chain = []
	theta_curr = prior_proposal()
	logL_curr = log_likelihood(data, theta_curr, dataset_range)
	logP_curr = log_prior(theta_curr)

	for i in pBar(range(0, MCMC_STEPS), pBarLabel):

		## pick which proposal
		theta_next = proposal_distribution(theta_curr)

		## compute likelihood and prior
		logL_next = log_likelihood(data, theta_next, dataset_range)
		logP_next = log_prior(theta_next)

		## compute Hasting's ratio
		logH =  logL_next - logL_curr + logP_next - logP_curr

		## draw acceptance probability
		logA = np.log(np.random.uniform(0,1))

		## Metropolis-Hasting's decision
		if logH > logA:
			theta_curr = theta_next
			logL_curr = logL_next
			logP_curr = logP_next
		chain.append(theta_curr)

	## remove burn-in
	del chain[:int(MCMC_STEPS/10)]
	chain = np.array(chain)

	return chain


def main():

	## open plot pdf
	pdf_image = PdfPages(os.path.join(os.getcwd(), 'MCMC.pdf'))

	#### read datasets ####

	## get dataset file names
	data_dir = os.path.join(os.getcwd(), 'fake_data')
	datasets_list = [data_file for data_file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, data_file))]
	datasets_list.sort()

	## get dataset data
	data = {}
	for i in range(len(datasets_list)):
		dataset_data = {}
		dataset = np.transpose(np.loadtxt(os.path.join(data_dir, datasets_list[i])))
		dataset_data['time'] = dataset[0]
		dataset_data['data'] = dataset[1]
		dataset_data['dFFT'] = np.fft.rfft(dataset[1], norm='ortho')
		data[i] = dataset_data

	#### MCMC for dataset 0 ####

	## run the MCMC
	chain0 = mcmc(data, (0,0), 'MCMC 0: ')

	## plot the chain samples for alpha & beta
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111)
	plt.plot(np.arange(len(chain0)), chain0[:,1], 'o')
	plt.plot(np.arange(len(chain0)), chain0[:,3], 'o')
	pdf_image.savefig(fig)

	## plot the chain samples for Sn_{alpha} & Sn_{beta}
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111)
	plt.plot(np.arange(len(chain0)), chain0[:,0], 'o')
	plt.plot(np.arange(len(chain0)), chain0[:,2], 'o')
	pdf_image.savefig(fig)

	## corner plot
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111)
	c = ChainConsumer()
	c.add_chain(chain0, parameters=[r'$S_\alpha$', r'$\alpha$', r'$S_\beta$', r'$\beta$'])
	fig = c.plotter.plot()
	pdf_image.savefig(fig)

	#### MCMC for datasets 1 and 2 ####

	## run the MCMC
	chain1 = mcmc(data, (1,1), 'MCMC 1: ')
	chain2 = mcmc(data, (2,2), 'MCMC 2: ')

	## corner plot
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111)
	c = ChainConsumer()
	c.add_chain(chain0, parameters=[r'$S_\alpha$', r'$\alpha$', r'$S_\beta$', r'$\beta$'])
	c.add_chain(chain1, parameters=[r'$S_\alpha$', r'$\alpha$', r'$S_\beta$', r'$\beta$'])
	c.add_chain(chain2, parameters=[r'$S_\alpha$', r'$\alpha$', r'$S_\beta$', r'$\beta$'])
	fig = c.plotter.plot()
	pdf_image.savefig(fig)

	#### MCMC for joint-analysis of datasets ####

	## run the joint-analysis MCMC
	chain_joint = mcmc(data, (0,2), 'Joint MCMC: ')

	#### compare the posteriors from the joint-analysis to the posterior from one of the individual analyses ####

	## corner plot
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111)
	c = ChainConsumer()
	c.add_chain(chain0, parameters=[r'$S_\alpha$', r'$\alpha$', r'$S_\beta$', r'$\beta$'])
	c.add_chain(chain_joint, parameters=[r'$S_\alpha$', r'$\alpha$', r'$S_\beta$', r'$\beta$'])
	fig = c.plotter.plot()
	pdf_image.savefig(fig)

	## close plot pdf
	pdf_image.close()

	return


if __name__ == "__main__":

    main()
