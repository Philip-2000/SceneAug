

class labl():
    def __init__(self,f="ful",n=None,i=None,c=None):
        self.field = f
        if n is not None: self.n = n
        elif i is not None: self.n = TRANSFER.fields[self.field].labls[i]
        elif c is not None: self.n = TRANSFER.fields[self.field].labls[c.argmax()]
        else: raise Exception("No value given")

    @property
    def i(self):
        return TRANSFER.fields[self.field].labls.index(self.n)
    
    @property
    def c(self):
        import numpy as np
        return np.eye(len(TRANSFER.fields[self.field]))[self.i]

    def __eq__(self, LABL):
        return TRANSFER(LABL,self.field) == self.n or TRANSFER(self,LABL.field) == LABL.n
        r = (TRANSFER(LABL,self.field) == self.n or TRANSFER(self,LABL.field) == LABL.n)
        if r: raise AssertionError("successfully transfer? %s %s =?= %s %s"%(self.field,self.n,LABL.field,LABL.n))
        return r
    
    def __call__(self,field):
        return TRANSFER(self,field)
    
    def __str__(self):
        return self.n
    
    def reverse(self,field):
        ret = []
        mapping = TRANSFER.fields[field].mappings[self.field]
        for k,v in mapping.items():
            if v == self.n: ret.append(k)
        return ret

        
class labls():
    def __init__(self):
        self.field = ""
        self.labls = []
        self.mappings = {}

    def __len__(self):
        return len(self.labls)
    
    def __getitem__(self,n):
        return labl(n=n,f=self.field)

    def __call__(self,labl):
        if labl.field == self.field: return labl.n #if the field is the same, we can directly return the value
        if not labl.field in self.mappings.keys(): return None #if the field is not in the mappings, we can't build transfer toward to it
        if not labl.n in self.mappings[labl.field].keys(): return None #if the value is not in the mappings, we can't map that value from that field to mine
        return self.mappings[labl.field][labl.n] #everything is fine, we can return the value

class labls_ful(labls):
    def __init__(self):
        super(labls_ful,self).__init__()
        self.field = "ful"
        self.labls =["Pendant Lamp", "Ceiling Lamp", "Bookcase / jewelry Armoire", \
                        "Round End Table", "Dining Table", "Sideboard / Side Cabinet / Console table", "Corner/Side Table", "Desk", "Coffee Table", "Dressing Table", \
                        "Children Cabinet", "Drawer Chest / Corner cabinet", "Shelf", "Wine Cabinet", \
                        "Lounge Chair / Cafe Chair / Office Chair", "Classic Chinese Chair", "Dressing Chair", "Dining Chair", "armchair", "Barstool", "Footstool / Sofastool / Bed End Stool / Stool", \
                        "Three-seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa", "Lazy Sofa", "Chaise Longue Sofa", "Wardrobe", "TV Stand", "Nightstand", \
                        "King-size Bed", "Kids Bed", "Bunk Bed", "Single bed", "Bed Frame", "window", "door"
                    ]
        self.mappings = {
            "bed":{
                "Desk":                                    "desk",
                "Nightstand":                              "nightstand",
                "King-size Bed":                           "double_bed",
                "Single bed":                              "single_bed",
                "Kids Bed":                                "kids_bed",
                "Ceiling Lamp":                            "ceiling_lamp",
                "Pendant Lamp":                            "pendant_lamp",
                "Bookcase / jewelry Armoire":              "bookshelf",
                "TV Stand":                                "tv_stand",
                "Wardrobe":                                "wardrobe",
                "Lounge Chair / Cafe Chair / Office Chair":"chair",
                "Dining Chair":                            "chair",
                "Classic Chinese Chair":                   "chair",
                "armchair":                                "armchair",
                "Dressing Table":                          "dressing_table",
                "Dressing Chair":                          "dressing_chair",
                "Corner/Side Table":                       "table",
                "Dining Table":                            "table",
                "Round End Table":                         "table",
                "Drawer Chest / Corner cabinet":           "cabinet",
                "Sideboard / Side Cabinet / Console table":"cabinet",
                "Children Cabinet":                        "children_cabinet",
                "Shelf":                                   "shelf",
                "Footstool / Sofastool / Bed End Stool / Stool": "stool",
                "Coffee Table":                            "coffee_table",
                "Loveseat Sofa":                           "sofa",
                "Three-seat / Multi-seat Sofa":              "sofa",
                "L-shaped Sofa":                           "sofa",
                "Lazy Sofa":                               "sofa",
                "Chaise Longue Sofa":                      "sofa",
            },
            "liv":{
                "Bookcase / jewelry Armoire":              "bookshelf",
                "Desk":                                    "desk",
                "Pendant Lamp":                            "pendant_lamp",
                "Ceiling Lamp":                            "ceiling_lamp",
                "Lounge Chair / Cafe Chair / Office Chair":"lounge_chair",
                "Dining Chair":                            "dining_chair",
                "Dining Table":                            "dining_table",
                "Corner/Side Table":                       "corner_side_table",
                "Classic Chinese Chair":                   "chinese_chair",
                "armchair":                                "armchair",
                "Shelf":                                   "shelf",
                "Sideboard / Side Cabinet / Console table":"console_table",
                "Footstool / Sofastool / Bed End Stool / Stool": "stool",
                "Barstool":                                "stool",
                "Round End Table":                         "round_end_table",
                "Loveseat Sofa":                           "loveseat_sofa",
                "Drawer Chest / Corner cabinet":           "cabinet",
                "Wardrobe":                                "wardrobe",
                "Three-seat / Multi-seat Sofa":            "multi_seat_sofa",
                "Wine Cabinet":                            "wine_cabinet",
                "Coffee Table":                            "coffee_table",
                "Lazy Sofa":                               "lazy_sofa",
                "Children Cabinet":                        "cabinet",
                "Chaise Longue Sofa":                      "chaise_longue_sofa",
                "L-shaped Sofa":                           "l_shaped_sofa",
                "TV Stand":                                "tv_stand"
            },
            "mrg":{
                "Bunk Bed":                                 "Bunk Bed", 
                "Bed Frame":                                "King-size Bed", 
                "Single bed":                               "King-size Bed", 
                "Kids Bed":                                 "King-size Bed",
                "Loveseat Sofa":                            "Three-seat / Multi-seat Sofa",
                "Lounge Chair / Cafe Chair / Office Chair": "Dining Chair", 
                "Classic Chinese Chair":                    "Dining Chair", 
                "Dressing Chair":                           "Dining Chair", 
                "armchair":                                 "Dining Chair",
                "Corner/Side Table":                        "Nightstand",
                "Ceiling Lamp":                             "Pendant Lamp",
                "Nightstand":                              "Nightstand",
                "Bookcase / jewelry Armoire":              "Bookcase / jewelry Armoire",
                "Desk":                                    "Desk",
                "Pendant Lamp":                            "Pendant Lamp",
                "Dining Chair":                            "Dining Chair", 
                "Dining Table":                            "Dining Table",
                "Dressing Table":                          "Dressing Table",
                "Shelf":                                   "Shelf",
                "Sideboard / Side Cabinet / Console table":"Sideboard / Side Cabinet / Console table",
                "Footstool / Sofastool / Bed End Stool / Stool": "Footstool / Sofastool / Bed End Stool / Stool",
                "Barstool":                                "Barstool",
                "Round End Table":                         "Round End Table",
                "King-size Bed":                           "King-size Bed",
                "Drawer Chest / Corner cabinet":           "Drawer Chest / Corner cabinet",
                "Wardrobe":                                "Wardrobe",
                "Three-seat / Multi-seat Sofa":            "Three-seat / Multi-seat Sofa",
                "Wine Cabinet":                            "Wine Cabinet",
                "Coffee Table":                            "Coffee Table",
                "Lazy Sofa":                               "Lazy Sofa",
                "Children Cabinet":                        "Children Cabinet",
                "Chaise Longue Sofa":                      "Chaise Longue Sofa",
                "L-shaped Sofa":                           "L-shaped Sofa",
                "TV Stand":                                "TV Stand"
            }
        }
    
class labls_mrg(labls):
    def __init__(self):
        super(labls_mrg,self).__init__()
        self.field = "mrg"
        self.labls = []
        self.mappings = {
            "ful":{
                "Nightstand":                              "Nightstand",
                "Bookcase / jewelry Armoire":              "Bookcase / jewelry Armoire",
                "Desk":                                    "Desk",
                "Pendant Lamp":                            "Pendant Lamp",
                "Dining Chair":                            "Dining Chair", 
                "Dressing Table":                          "Dressing Table",
                "Dining Table":                            "Dining Table",
                "Shelf":                                   "Shelf",
                "Sideboard / Side Cabinet / Console table":"Sideboard / Side Cabinet / Console table",
                "Footstool / Sofastool / Bed End Stool / Stool": "Footstool / Sofastool / Bed End Stool / Stool",
                "Barstool":                                "Barstool",
                "Round End Table":                         "Round End Table",
                "King-size Bed":                           "King-size Bed",
                "Drawer Chest / Corner cabinet":           "Drawer Chest / Corner cabinet",
                "Wardrobe":                                "Wardrobe",
                "Three-seat / Multi-seat Sofa":            "Three-seat / Multi-seat Sofa",
                "Wine Cabinet":                            "Wine Cabinet",
                "Coffee Table":                            "Coffee Table",
                "Lazy Sofa":                               "Lazy Sofa",
                "Children Cabinet":                        "Children Cabinet",
                "Chaise Longue Sofa":                      "Chaise Longue Sofa",
                "L-shaped Sofa":                           "L-shaped Sofa",
                "TV Stand":                                "TV Stand"
            }
        }

class labls_bed(labls):
    def __init__(self):
        super(labls_bed,self).__init__()
        self.field = "bed"
        self.labls = [
            "armchair",
            "bookshelf",
            "cabinet",
            "ceiling_lamp",
            "chair",
            "children_cabinet",
            "coffee_table",
            "desk",
            "double_bed",
            "dressing_chair",
            "dressing_table",
            "kids_bed",
            "nightstand",
            "pendant_lamp",
            "shelf",
            "single_bed",
            "sofa",
            "stool",
            "table",
            "tv_stand",
            "wardrobe",
        ]
        self.mappings = {
            "ful":{
                "desk":                                    "Desk",
                "nightstand":                              "Nightstand",
                "double_bed":                              "King-size Bed",
                "single_bed":                              "Single bed",
                "kids_bed":                                "Kids Bed",
                "ceiling_lamp":                            "Ceiling Lamp",
                "pendant_lamp":                            "Pendant Lamp",
                "bookshelf":                               "Bookcase / jewelry Armoire",
                "tv_stand":                                "TV Stand",
                "wardrobe":                                "Wardrobe",
                "chair":                                   "Lounge Chair / Cafe Chair / Office Chair",
                "armchair":                                "armchair",
                "dressing_table":                          "Dressing Table",
                "dressing_chair":                          "Dressing Chair",
                "table":                                   "Corner/Side Table",
                "cabinet":                                 "Drawer Chest / Corner cabinet",
                "children_cabinet":                        "Children Cabinet",
                "shelf":                                   "Shelf",
                "stool":                                   "Footstool / Sofastool / Bed End Stool / Stool",
                "coffee_table":                            "Coffee Table",
                "sofa":                                    "Chaise Longue Sofa",
            },
        }

class labls_liv(labls):
    def __init__(self):
        super(labls_liv,self).__init__()
        self.field = "liv"
        self.labls = [
            "armchair",
            "bookshelf",
            "cabinet",
            "chaise_longue_sofa",
            "chinese_chair",
            "coffee_table",
            "console_table",
            "corner_side_table",
            "desk",
            "dining_chair",
            "dining_table",
            "l_shaped_sofa",
            "lazy_sofa",
            "lounge_chair",
            "loveseat_sofa",
            "multi_seat_sofa",
            "round_end_table",
            "shelf",
            "stool",
            "tv_stand",
            "wardrobe",
            "wine_cabinet",
        ]
        self.mappings = {
            "ful":{
                "bookshelf":                               "Bookcase / jewelry Armoire",
                "desk":                                    "Desk",
                "pendant_lamp":                            "Pendant Lamp",
                "ceiling_lamp":                            "Ceiling Lamp",
                "lounge_chair":                            "Lounge Chair / Cafe Chair / Office Chair",
                "dining_chair":                            "Dining Chair",
                "dining_table":                            "Dining Table",
                "corner_side_table":                       "Corner/Side Table",
                "chinese_chair":                           "Classic Chinese Chair",
                "armchair":                                "armchair",
                "shelf":                                   "Shelf",
                "console_table":                           "Sideboard / Side Cabinet / Console table",
                "stool":                                   "Footstool / Sofastool / Bed End Stool / Stool",
                "round_end_table":                         "Round End Table",
                "loveseat_sofa":                           "Loveseat Sofa",
                "cabinet":                                 "Drawer Chest / Corner cabinet",
                "wardrobe":                                "Wardrobe",
                "multi_seat_sofa":                         "Three-seat / Multi-seat Sofa",
                "wine_cabinet":                            "Wine Cabinet",
                "coffee_table":                            "Coffee Table",
                "lazy_sofa":                               "Lazy Sofa",
                "cabinet":                                 "Children Cabinet",
                "chaise_longue_sofa":                      "Chaise Longue Sofa",
                "l_shaped_sofa":                           "L-shaped Sofa",
                "tv_stand":                                "TV Stand"
            },
        }

class labls_clp(labls):
    def __init__(self):
        super(labls_clp,self).__init__()
        self.field = "clp"
        self.labls = []
        self.mappings = {}

class transfer():
    def __init__(self):
        self.fields = {
            "ful":labls_ful(),
            "mrg":labls_mrg(),
            "bed":labls_bed(),
            "liv":labls_liv(),
            "clp":labls_clp(),
        }
    
    def __call__(self,labl,field):
        return self.fields[labl.field].mappings[field][labl.n]
    
TRANSFER = transfer()