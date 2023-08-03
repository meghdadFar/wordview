# import nltk
# from patterns import ENPatterns, DEPatterns

# class MWEChunker:
#     def __init__(self, language='EN'):
#         self.language = language.upper()
#         self.chunk_parser = None

#         if self.language == 'EN':
#             all_en_patterns = '\n'.join(EN_LIGHT_VERB_PATTERNS + EN_NOUN_COMPOUND_PATTERNS +
#                                      EN_ADJ_NOUN_COMPOUND_PATTERNS + EN_VERB_PARTICLE_PATTERNS)

#             for _, value in ENPatterns.patterns.items():
#                 all_en_patterns += '\n'.join(value)
#             # Create the RegexpParser with all patterns
#             self.chunk_parser = nltk.RegexpParser(all_en_patterns)

#         else:
#             raise ValueError("Language not supported. Use 'EN' for English.")

#     def parse(self, sentence):
#         # Tokenize the sentence
#         tokens = nltk.word_tokenize(sentence)

#         # Part-of-speech tagging
#         pos_tags = nltk.pos_tag(tokens)

#         # Use the tagged sentence for parsing
#         parsed_tree = self.chunk_parser.parse(pos_tags)

#         return parsed_tree


# # Example usage
# if __name__ == "__main__":
#     custom_chunker_en = MWEChunker(language='EN')

#     sentence_en = "I will take a walk and give a speech. The coffee shop near the swimming pool sells red apples."

#     parsed_tree_en = custom_chunker_en.parse(sentence_en)

#     # Print the parsed tree
#     print(parsed_tree_en)
