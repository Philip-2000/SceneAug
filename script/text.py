def parse(): #using pattern to synthesis scene
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Text')
    
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__": 
    args = parse()
    from SceneClasses.Semantic import text
    T = text(["The scene has a King-size Bed, a Wardrobe and a Nightstand",
                    " the Wardrobe is on the left of  the King-size Bed"])
    from SceneClasses.Basic import scne
    from SceneClasses.Operation.Syth import gnrt
    from SceneClasses.Operation.Patn import patternManager as pm
    P = pm("losy")
    S = scne.empty("text",keepEmptyWL=True)
    S.TEXTS = T
    G = gnrt(P,S)
    G.textcond(draw=True)
