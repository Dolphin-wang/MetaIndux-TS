## Necessary Packages
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def display_scores(results):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
  #  sigma = 1.96*(np.std(results)/np.sqrt(len(results)))
   print('Final Score: ', f'{mean} \xB1 {sigma}')


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def visualization(ori_data, generated_data, analysis, compare=3000):
    """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca or kernel
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min(compare, ori_data.shape[0], generated_data.shape[0])
    print(anal_sample_no)
    if ori_data.shape[0] < generated_data.shape[0]:
        idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]
    else:
        idx = np.random.permutation(generated_data.shape[0])[:anal_sample_no]

    # Data preprocessing
    # ori_data = np.asarray(ori_data)
    # generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c="red", alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c="blue", alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.show()

    elif analysis == 'tsne':
      ori_sample_no = min(compare, ori_data.shape[0])
      gen_sample_no = min(compare, generated_data.shape[0])

      ori_idx = np.random.permutation(ori_data.shape[0])[:ori_sample_no]
      gen_idx = np.random.permutation(generated_data.shape[0])[:gen_sample_no]

      ori_data = ori_data[ori_idx]
      generated_data = generated_data[gen_idx]

      no, seq_len, dim = ori_data.shape  

      prep_data = np.array([np.mean(ori_data[i, :, :], axis=1) for i in range(ori_sample_no)])
      prep_data_hat = np.array([np.mean(generated_data[i, :, :], axis=1) for i in range(gen_sample_no)])
      prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

      tsne = TSNE(n_components=2, perplexity=40, n_iter=300, verbose=1)
      tsne_results = tsne.fit_transform(prep_data_final)

      plt.figure(figsize=(8, 6))
      plt.scatter(tsne_results[:ori_sample_no, 0], tsne_results[:ori_sample_no, 1], 
                  c="red", alpha=0.5, label="Original")
      plt.scatter(tsne_results[ori_sample_no:ori_sample_no + gen_sample_no, 0], 
                  tsne_results[ori_sample_no:ori_sample_no + gen_sample_no, 1], 
                  c="blue", alpha=0.5, label="Synthetic")

      plt.xlim(-15, 15)  
      plt.ylim(-7, 7)  

      plt.legend()
      plt.title('t-SNE Visualization')
      plt.xlabel('t-SNE Component 1')
      plt.ylabel('t-SNE Component 2')
      plt.show()
    elif analysis == 'kernel':
       
        # Visualization parameter
        # colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

        f, ax = plt.subplots(1)
        sns.distplot(prep_data, hist=False, kde=True, kde_kws={'linewidth': 5}, label='Original', color="red")
        sns.distplot(prep_data_hat, hist=False, kde=True, kde_kws={'linewidth': 5, 'linestyle':'--'}, label='Synthetic', color="blue")
        # Plot formatting

        # plt.legend(prop={'size': 22})
        plt.legend()
        plt.xlabel('Data Value')
        plt.ylabel('Data Density Estimate')
        # plt.rcParams['pdf.fonttype'] = 42

        # plt.savefig(str(args.save_dir)+"/"+args.model1+"_histo.png", dpi=100,bbox_inches='tight')
        # plt.ylim((0, 12))
        plt.show()
        plt.close()


def visualization_zeroshot(ori_data, generated_data, extra_data, analysis, compare=500):
    """Using PCA, t-SNE, or Kernel Density Estimation (KDE) for visualization of three datasets.
  
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - extra_data: third dataset to be visualized
        - analysis: 'tsne', 'pca', or 'kernel'
    """
    
    ori_sample_no = min(4 * compare, ori_data.shape[0])
    gen_sample_no = min(3 * compare, generated_data.shape[0])
    extra_sample_no = min(1 * compare, extra_data.shape[0])

    ori_idx = np.random.permutation(ori_data.shape[0])[:ori_sample_no]
    gen_idx = np.random.permutation(generated_data.shape[0])[:gen_sample_no]
    extra_idx = np.random.permutation(extra_data.shape[0])[:extra_sample_no]

    ori_data = ori_data[ori_idx]
    generated_data = generated_data[gen_idx]
    extra_data = extra_data[extra_idx]

    no, seq_len, dim = ori_data.shape

    prep_data = np.array([np.mean(ori_data[i, :, :], axis=1) for i in range(ori_sample_no)])
    prep_data_hat = np.array([np.mean(generated_data[i, :, :], axis=1) for i in range(gen_sample_no)])
    prep_extra = np.array([np.mean(extra_data[i, :, :], axis=1) for i in range(extra_sample_no)])
    colors = (["red"] * ori_sample_no) + (["blue"] * gen_sample_no) + (["green"] * extra_sample_no)

    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        pca_extra_results = pca.transform(prep_extra)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c="red", alpha=0.5, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c="blue", alpha=0.5, label=">30 Data")
        plt.scatter(pca_extra_results[:, 0], pca_extra_results[:, 1], c="green", alpha=0.5, label="<30 Data")
        
        plt.legend()
        plt.title('PCA Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

    elif analysis == 'tsne':
      prep_data_final = np.concatenate((prep_data, prep_data_hat, prep_extra), axis=0)

      tsne = TSNE(n_components=2, perplexity=40, n_iter=300, verbose=1)
      tsne_results = tsne.fit_transform(prep_data_final)

      plt.figure(figsize=(8, 6))
      plt.scatter(tsne_results[:ori_sample_no, 0], tsne_results[:ori_sample_no, 1], 
                  c="red", alpha=0.5, label="Full lifecycle")
      plt.scatter(tsne_results[ori_sample_no:ori_sample_no + gen_sample_no, 0], 
                  tsne_results[ori_sample_no:ori_sample_no + gen_sample_no, 1], 
                  c="blue", alpha=0.5, label="Syn. Normal")
      plt.scatter(tsne_results[ori_sample_no + gen_sample_no:, 0], 
                  tsne_results[ori_sample_no + gen_sample_no:, 1], 
                  c="green", alpha=0.5, label="Syn. Fatigue")


      plt.legend(fontsize=20)
      plt.title('t-SNE plot', fontsize=24) 
      plt.gca().set_xlabel('') 
      plt.gca().set_ylabel('')  
      plt.gca().get_xaxis().set_visible(False)  
      plt.gca().get_yaxis().set_visible(False)  
      #plt.show()

      plt.figure(figsize=(8, 6))
      plt.scatter(tsne_results[:ori_sample_no, 0], tsne_results[:ori_sample_no, 1], 
                  c="red", alpha=0.5, label="Full lifecycle")
      plt.scatter(tsne_results[ori_sample_no:ori_sample_no + gen_sample_no, 0], 
                  tsne_results[ori_sample_no:ori_sample_no + gen_sample_no, 1], 
                  c="blue", alpha=0.5, label="Syn. Normal")

      plt.legend(fontsize=20)
      plt.title('t-SNE plot', fontsize=24)  
      plt.gca().set_xlabel('')  
      plt.gca().set_ylabel('')  
      plt.gca().get_xaxis().set_visible(False)  
      plt.gca().get_yaxis().set_visible(False)  

      plt.show()

    elif analysis == 'kernel':
        
        plt.figure(figsize=(8, 6))
        sns.kdeplot(prep_data.flatten(), label="Original", color="red", linewidth=2)
        sns.kdeplot(prep_data_hat.flatten(), label="Synthetic", color="blue", linestyle="--", linewidth=2)
        sns.kdeplot(prep_extra.flatten(), label="Extra Data", color="green", linestyle=":", linewidth=2)

        plt.legend()
        plt.xlabel('Data Value')
        plt.ylabel('Density')
        plt.title('Kernel Density Estimation (KDE)')
        plt.show()
        plt.close()


if __name__ == '__main__':
   pass