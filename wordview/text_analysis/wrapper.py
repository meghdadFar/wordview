from typing import Set
from wordview.text_analysis.core import do_txt_analysis, plotly_wordcloud
import pandas
import plotly.graph_objs as go
import plotly.figure_factory as ff


class TextStatsPlots:
    def __init__(self,
        df: pandas.DataFrame,
        text_column: str,
        distributions: Set=['doc_len', 'word_frequency_zipf'],
        pos_tags: Set=['NN', 'VB', 'JJ']
        ) -> None:
        self.df = df
        self.analysis = do_txt_analysis(df=self.df,
                               text_col=text_column)
        self.distributions = distributions
        self.dist_plots = self.create_dist_plots()
        self.pos_tags = pos_tags
        self.pos_plots = self.create_pos_plots()
        self.languages = self.analysis.languages
        self.type_count = self.analysis.type_count
        self.token_count = self.analysis.token_count
        self.num_docs = self.analysis.doc_count
        self.median_doc_len = self.analysis.median_doc_len
        self.num_nns = len(self.analysis.nns)
        self.num_jjs = len(self.analysis.jjs)
        self.num_vbs = len(self.analysis.vs)

    def create_dist_plots(self):
        """Creates distribution plots for items in `self.distributions`."""
        dist_plot_setup = {
                    'paper_bgcolor': '#007A78',
                    }
        res = {}
        if 'doc_len' in self.distributions:
            fig_doc_len_dist = ff.create_distplot([self.analysis.doc_lengths], group_labels=["distplot"], colors=["blue"])
            res['doc_len'] = fig_doc_len_dist 

        if 'word_frequency_zipf' in self.distributions:
            fig_w_freq = go.Figure()
            fig_w_freq.add_trace(go.Scattergl(x=self.analysis.zipf_x, y=self.analysis.zipf_y_emp, mode='markers'))
            fig_w_freq.add_trace(go.Scattergl(x=self.analysis.zipf_x, y=self.analysis.zipf_y_theory, mode='markers'))
            res['word_frequency_zipf'] = fig_w_freq
        
        for _, fig in res.items():
            fig.update_layout(dist_plot_setup)

        return res

    def create_pos_plots(self):
        word_cloud_setup = {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                            'xaxis_showgrid':False,
                            'yaxis_showgrid':False,
                            'xaxis_zeroline':False,
                            'yaxis_zeroline':False,
                            'yaxis_visible':False,
                            'yaxis_showticklabels':False,
                            'xaxis_visible':False,
                            'xaxis_showticklabels':False,
                            }
        res = {}
        if 'NN' in self.pos_tags:
            res['noun_cloud'] = go.Figure(plotly_wordcloud(token_count_dic=self.analysis.nns))
        if 'JJ' in self.pos_tags:
            res['adj_cloud'] = go.Figure(plotly_wordcloud(token_count_dic=self.analysis.jjs))
        if 'VB' in self.pos_tags:
            res['verb_cloud'] = go.Figure(plotly_wordcloud(token_count_dic=self.analysis.vs))
            
        for _, fig in res.items():
            fig.update_layout(word_cloud_setup)
        
        return res