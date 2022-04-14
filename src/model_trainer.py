from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from scipy.special import expit

def visualize_topics(model, vocab, num_topics, num_words=10):
	model.eval() 
	with torch.no_grad():
		print('#'*100)
		print('Visualize topics...')
		betas = model.alphas.t()
		for k in range(num_topics):
			beta = betas[k].detach().numpy()
			top_words = beta.argsort()[-num_words:]
			topic_words = vocab[top_words]
			print('Topic {}: {}'.format(k, topic_words))

def get_representation(model, docs):
	normalized = docs/docs.sum(axis=-1)[:,np.newaxis]
	normalized_bow = torch.tensor(normalized, dtype=torch.float)
	num_documents = docs.shape[0]
	model.eval()
	with torch.no_grad():
		doc_representation,_ = model.get_theta(normalized_bow)
		embeddings = doc_representation.detach().numpy()
	return embeddings


def predict(model, docs, is_label_binary=False):
	normalized = docs/docs.sum(axis=-1)[:,np.newaxis]
	normalized_bow = torch.tensor(normalized, dtype=torch.float)
	num_documents = docs.shape[0]

	treatment_ones = torch.ones(num_documents) 
	treatment_zeros = torch.zeros(num_documents) 

	model.eval()
	with torch.no_grad():
		doc_representation,_ = model.get_theta(normalized_bow)
		propensity_score = model.predict_treatment(doc_representation).squeeze().detach().numpy()
		propensity_score = expit(propensity_score)
		expected_outcome_treat = model.predict_outcome_st_treat(doc_representation, treatment_ones).squeeze().detach().numpy()
		expected_outcome_no_treat = model.predict_outcome_st_no_treat(doc_representation, treatment_zeros).squeeze().detach().numpy()

		if is_label_binary:
			expected_outcome_treat = expit(expected_outcome_treat)
			expected_outcome_no_treat = expit(expected_outcome_no_treat)
		
		return propensity_score, expected_outcome_treat, expected_outcome_no_treat

def train(model, docs, treatment_labels, outcomes, is_label_binary=False, num_iters=300, lr=0.005, wdecay=1.2e-6,batch_size=512, use_recon_loss=True, use_treatment_loss=True, use_outcome_loss=True):
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
	num_documents = docs.shape[0]
	indices = np.arange(num_documents)
	np.random.shuffle(indices)

	for e_idx in range(num_iters):
		model.train()
		k = e_idx%(num_documents//batch_size)
		start_index = k*batch_size
		end_index = (k+1)*batch_size
		batch = indices[start_index:end_index]
		docs_batch = docs[batch,:]
		treatment_labels_batch = treatment_labels[batch]
		outcomes_batch = outcomes[batch]
		normalized_batch = docs_batch/docs_batch.sum(axis=1)[:,np.newaxis]
		
		outcome_labels = torch.tensor(outcomes_batch, dtype=torch.float)
		treat_labels = torch.tensor(treatment_labels_batch, dtype=torch.float)
		bow = torch.tensor(docs_batch, dtype=torch.float)
		normalized_bow = torch.tensor(normalized_batch, dtype=torch.float)

		optimizer.zero_grad()
		model.zero_grad()

		recon_loss, outcome_loss, treatment_loss, kld_theta = model(bow, 
			normalized_bow, treat_labels, 
			outcome_labels, is_label_binary=is_label_binary,
			use_outcome_loss=use_outcome_loss, use_treatment_loss=use_treatment_loss)

		acc_kl_theta_loss = torch.sum(kld_theta).item()
		acc_out_loss = 0.
		acc_treat_loss = 0.
		acc_loss = 0.
		
		total_loss = kld_theta
		if use_recon_loss:
			acc_loss = torch.sum(recon_loss).item()
			total_loss += recon_loss
		if use_treatment_loss:
			acc_treat_loss = torch.sum(treatment_loss).item()
			total_loss += treatment_loss
		if use_outcome_loss:
			acc_out_loss = torch.sum(outcome_loss).item()
			total_loss += outcome_loss

		total_loss.backward()
		optimizer.step()
		
		if e_idx % 10 == 0:
			print("Acc. loss:", acc_loss, "KL loss.:", acc_kl_theta_loss, "Outcome loss:", acc_out_loss, "Treatment loss:", acc_treat_loss)