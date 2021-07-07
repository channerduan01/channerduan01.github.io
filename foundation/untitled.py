#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:14:21 2019

@author: channerduan
"""

import jieba
import re

class WordSeparator:
    # 初始化领域词表
    def __init__(self, extra_word_pair_dict_path = None):
        print("WordSeparator init...")

        dict_list = [
	        ("product_names", "支付宝,余额宝,网商贷,花呗,借呗,余利宝,相互宝,天猫", 999),
			("promo_words", "红包,福利,优惠,立减,礼包,特惠,抽奖,好礼,体验金", 666),
			("action_words", "点击,付款,转账,支付,到店付款,收藏,许愿,答题,收取", 666),
			("donate_words", "公益,森林,小鸡,能量,饲料", 666),
			("promo_desc_words", "有料,速抢,必抢,网红,劲爆,最高价值,不限量,可叠加,可抵用", 333),
		]

		word_property_dict = {}
		for dict_ in dict_list:
			for s in dict_[1]:
				jieba.add_word(s, dict_[2])
				word_property_dict[s] = dict_[0]
        
    # Main entrance function for class level
    def sepa(self, text):
        raw_sepa_list = WordSeparator.__ban_word_detect(jieba.lcut(text))
        tag_sepa_list = WordSeparator.__tag_word_concate(raw_sepa_list)
        concat_list = WordSeparator.__concat_pair_detect(tag_sepa_list)
        return concat_list

    
    # BAN WORDS
    BAN_CONF = [' ']
    @staticmethod
    def __ban_word_detect(text):
        return filter(lambda x: x not in WordSeparator.BAN_CONF, text)
    
    # TAG DETECTOR
    TAG_DETECTED_CONF = [(u'<', [u'>']), (u'[', [u']', u'】']), (u'【', [u']', u'】']), (u'〔', [u'〕']), (u'〖', [u'〗']), (u'(', [u')', u'）']), (u'（', [u'）', u')']), (u'\'', [ u'\'']), (u'‘', [u'’']), (u'"', [u'"']), (u'“', [u'”']), (u'「', [u'」']), (u'{', [u'}']), (u'｛', [u'｝'])]
    TAG_INDEX_DICT = dict(zip(map(lambda x: x[0], TAG_DETECTED_CONF),range(len(TAG_DETECTED_CONF))))
    @staticmethod
    def __tag_word_concate(in_list):
        out_list = []
        tag_index = -1
        cache_list = []
        for s in in_list:
            if tag_index == -1:
                if s in WordSeparator.TAG_INDEX_DICT:
                    tag_index = WordSeparator.TAG_INDEX_DICT[s]
                else:
                    out_list.append(s)
            if tag_index > -1:
                cache_list.append(s)
                if s in WordSeparator.TAG_DETECTED_CONF[tag_index][1]:
                    tag_index = -1
                    out_list.append(''.join(cache_list))
                    cache_list = []
        out_list.extend(cache_list)
        return out_list
    
    @staticmethod
    def is_tag_word(text):
        if text is None or len(text) == 0:
            return False
        if text[0] in WordSeparator.TAG_INDEX_DICT:
            if text[-1] in WordSeparator.TAG_DETECTED_CONF[WordSeparator.TAG_INDEX_DICT[text[0]]][1]:
                return True
        return False
    
    @staticmethod
    def __concat_pair_detect(in_list):
        out_list = []
        if len(in_list) == 0:
            return out_list
        merged_flag = False
        for i in range(len(in_list)-1):
            if merged_flag:
                merged_flag = False
                continue
            s1 = in_list[i]
            s2 = in_list[i+1]
            if WordSeparator.__is_number(s1):
                is_matched = False
                for ss in [u'折', u'万', u'个', u'元', u'只', u'份']:
                    if s2.startswith(ss):
                        out_list.append('%s%s' %(s1, ss))
                        in_list[i+1] = in_list[i+1][1:]
                        is_matched = True
                        break
                if not is_matched:
                    out_list.append(s1)
            else:
                out_list.append(s1)
        if not merged_flag:
            out_list.append(in_list[-1])
        return out_list
    
    @staticmethod
    def __is_number(num):
      pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
      result = pattern.match(num)
      if result:
        return True
      else:
        return False
    

