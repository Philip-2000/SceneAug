class text_obj():
    def __init__(self, text, idx, origin):
        self.text = text
        #orders = {"first":0,"second":1,"third":2,"fourth":3,"fifth":4,"sixth":5,"seventh":6,"eighth":7,"ninth":8,"tenth":9}
        redros = {0:"first",1:"second",2:"third",3:"fourth",4:"fifth",5:"sixth",6:"seventh",7:"eighth",8:"ninth",9:"tenth"}
        self.desc = redros[idx]
        self.idx = idx
        self.origin = origin
        self.oid = None
        self.nid = -1
    
    def __str__(self):
        return "(%d)%s"%(self.idx, self.text) + \
                (" o(%d)"%self.oid if self.oid is not None else "") + (" n(%d)"%self.nid if self.nid != -1 else "")

    @property
    def flags(self):
        return "(%d)%s"%(self.idx, self.text)

class text_rel():
    def __init__(self, sen, origin):
        formatStr = [" is on the left of ", " is on the right of ", " is in front of ", " is behind ", " is close to "]
        try:
            fms = [f for f in formatStr if sen.find(f) != -1][0]
        except:
            raise Exception("No format string found in the sentence.")
        
        self.a = origin.search_obj(sen[:sen.find(fms)])
        self.f = fms
        self.b = origin.search_obj(sen[sen.find(fms)+len(fms):])
        #描述符之间的关系

    def check(self, oa, ob, pm=None):
        if pm is not None:
            from ..Basic import obje
            A = obje.empty(n=self.a.text)
            A.nid = oa
            B = pm.rel_object(A, ob)
        else:
            A,B = oa,ob
        B_A = A - B 
        if self.f  == " is on the left of ": return B_A.translation[0] < 0
        elif self.f== " is on the right of ":return B_A.translation[0] > 0
        elif self.f== " is in front of ":    return B_A.translation[2] < 0
        elif self.f== " is behind ":         return B_A.translation[2] > 0
        elif self.f== " is close to ":       return (B.translation - A.translation)**2 < 1

    def __str__(self):
        simp = {" is on the left of ":"left", " is on the right of ":"right", " is in front of ":"front", " is behind ":"behind", " is close to ":"close"}
        return "(%d)%s ->%s-> (%d)%s"%(self.a.idx, self.a.text, simp[self.f], self.b.idx, self.b.text) 

class text():
    def __init__(self, sens):
        self.objs= {}
        self.rels = []
        self.parses(sens)
    
    def objTypeForm(self, text):
        return text.strip()
    
    def pronoun_obj(self, text):
        pronouns = [" a "," one "," the "]
        for p in pronouns:
            if p in text: text = text.replace(p,"")
        pronouns = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        ps = [p for p in pronouns if p in text]
        assert len(ps) <= 1
        if len(ps) == 0:
            return None, self.objTypeForm(text.strip())
        else:
            text = text.replace(ps[0],"")
            return pronouns.index(ps[0]), self.objTypeForm(text.strip())

    def num_obj(self, text):
        text = text.strip()
        #print("num_obj",text,(" ".join(text.split(" ")[1:])))
        nums = {"a":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        for k in nums:
            if k == text.split(" ")[0]:
                return nums[k], self.objTypeForm((" ".join(text.split(" ")[1:])))
        return 1, self.objTypeForm(text.strip())
        #return nums[text.strip().split(" ")[0]], self.objTypeForm(text.strip().split(" ")[1])
    
    def search_obj(self, text):
        a,b = self.pronoun_obj(text)
        a = 0 if a is None else a
        if (a==0 and (b not in self.objs)) or len(self.objs[b]) <= a: self.add(b, a+1)
        return self.objs[b][a]
        
    def add(self, text, num=1):
        for i in range(num):
            if text in self.objs: self.objs[text].append(text_obj(text, len(self.objs[text]), self))
            else: self.objs[text] = [text_obj(text, 0, self)]
    
    def append(self, objes):
        cnts = {}
        for o in objes:
            if o.class_name in cnts: cnts[o.class_name] += 1
            else: cnts[o.class_name] = 1
        for t in cnts:
            if (not t in self.objs) or len(self.objs[t]) < cnts[t]: self.add(t, cnts[t]-len(self.objs[t]))

    def parse(self, sen):
        if sen.find("has") > -1:
            sen = sen[sen.find("has")+3:]
            if sen.find(" and ") > -1: sen = sen.replace(" and ",",")
            obs = sen.split(',')
            for ob in obs:
                ns = self.num_obj(ob) #ob.split(" ")
                self.add(ns[1], ns[0])
        else:
            self.rels.append(text_rel(sen, self))

    def parses(self, sens):
        for sen in sens: self.parse(sen)
    
    def __str__(self):
        return "\n".join([str(o) for k in self.objs for o in self.objs[k]]) + "\n" + "\n".join([str(r) for r in self.rels])

    @property
    def flags(self):
        from itertools import chain
        return list(chain(*[[o.flags for o in self.objs[k]] for k in self.objs]))

"""
def extract_relations(nlp,sentence):

    from spacy.matcher import Matcher

    matcher = Matcher(nlp.vocab)

    # 定义表示方位的常见复合介词短语
    patterns = [
        [{"dep": "prep", "lower": "in"}, {"dep": "pobj", "lower": "front"}, {"dep": "prep", "lower": "of"}],      # in front of
        [{"dep": "prep", "lower": "next"}, {"dep": "compound", "lower": "to"}, {"dep": "pobj", "tag": "NN"}],       # next to
        [{"dep": "prep", "lower": "on"}, {"dep": "pobj", "lower": "top"}, {"dep": "pobj", "tag": "NN"}],        # on top of
        [{"dep": "prep", "lower": "under"}, {"dep": "compound", "lower": "neath"}, {"dep": "pobj", "tag": "NN"}],   # underneath
        [{"dep": "prep", "lower": "behind"}, {"dep": "pobj", "tag": "NN"}],                                         # behind
        [{"dep": "prep", "lower": "in"}, {"dep": "compound", "lower": "between"}, {"dep": "pobj", "tag": "NN"}],    # in between
        [{"dep": "prep", "lower": "at"}, {"dep": "det", "lower": "the"}, {"dep": "pobj", "tag": "NN"}],             # at the side of
        [{"dep": "prep", "lower": "far"}, {"dep": "compound", "lower": "from"}, {"dep": "pobj", "tag": "NN"}],      # far from
        [{"dep": "prep", "lower": "close"}, {"dep": "to", "tag": "IN"}, {"dep": "pobj", "tag": "NN"}],              # close to
        [{"dep": "prep", "lower": "on"}, {"dep": "det", "lower": "the"}, {"dep": "left", "tag": "NN"}],             # on the left of
        [{"dep": "prep", "lower": "on"}, {"dep": "det", "lower": "the"}, {"dep": "right", "tag": "NN"}],            # on the right of
    ]

    # 添加模式到 Matcher
    for pattern in patterns:
        matcher.add("spatial_prepositions", [pattern])

    doc = nlp(sentence)
    edges = []

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]  # 获取匹配到的复合介词短语
        print(span[0].head.text, span.text, span[-1].text)
    return
    for token in doc:
        print(token,token.dep_,token.pos_,"\t",token.head.text,token.head.dep_,token.head.pos_,[child for child in token.children])
        # 处理介词短语（prep + pobj）
        if token.pos_ == "ADP":  # 介词 (e.g., in, on, under)
            prep_phrase = [token.text]  # 存储介词短语
            pobj = None  # 介词宾语

            # 查找后续的组成部分
            for child in token.children:
                if child.dep_ == "pobj":  # 介词宾语
                    pobj = child.text
                elif child.dep_ == "prep":  # 复合介词
                    prep_phrase.append(child.text)

            if pobj:
                full_prep = " ".join(prep_phrase)  # 形成完整的介词短语
                head = token.head.text  # 介词依赖的主语
                edges.append((head, full_prep, pobj))

        # 处理普通 SVO 结构
        if token.dep_ in ["nsubj", "dobj"]:
            subject = token.head.text if token.dep_ == "dobj" else token.text
            verb = token.head.text if token.dep_ == "nsubj" else token.text
            object_ = token.text if token.dep_ == "dobj" else token.head.text
            edges.append((subject, verb, object_))

    return edges

def draw_graph(edges):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()

    for subj, rel, obj in edges:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=rel)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))


    nx.draw(G, pos, ax=(plt.gcf()).add_axes((0, 0, 1, 1)), with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    edge_labels = {(subj, obj): rel for subj, rel, obj in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.show()

# 测试句子
if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm")
    sentence = "The book in front of the table and the cat is next to the chair."
    edges = extract_relations(nlp,sentence)
    print(edges)#draw_graph(edges)
"""
