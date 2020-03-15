import numpy as np
from skimage.transform import radon, iradon
import os


pathForSaving =  os.getcwd() + '/data/'
pathForLabel = os.getcwd() + '/label/'


part = 0    # Numbering for Ground Truth
cntr =  0   # Numbering for texture direction

winkelwerte = np.linspace(-100, 100, 25)
groundTruth = np.zeros((winkelwerte.size*100,)) 
for runTenTimes in range(0, 100):
    for k in range(0, winkelwerte.size):
        M = np.int( np.around(301*np.sqrt(2)) ) # = rho
        N = 2001
        thetabereich = 1
        theta = np.linspace( -thetabereich, thetabereich, N) # +(k-1)*5
        S = np.zeros( (M, N) )

        # Angle Area 1 (width of Radontrafo)
        sigmaw = 50
        muw = 1001 + winkelwerte[k]
        sigmawinkel = 10 # Variation der Riefenrichtung

        # rho (Distance of  Grooves)
        mua = 10
        sigmaa = 100
        mu2a = 5
        sigma2a = 50
        # Width of Grooves
        breitea = 5
        breiteb = 10
        breite2a = 20
        breite2b = 30

        # Set the depth of the grooves
        hoehe = 600

        # Run in M direction, i.e. in rho direction 
        m = np.around( 0 + 0 * np.random.rand(1, 1) )
        mend = 0 + 0 * np.random.rand(1,1)
        while m < M:
            # Breite der Rillen
            breite = np.round(breitea + (breiteb-breitea)*np.random.rand(1,1) )
            if breite < 1:
                breite = 1
            breite = np.int(breite)

            # Schrittweite in M-Richtung
            mstep = np.round( mua + sigmaa * np.random.rand(1,1))
            m = m + mstep
            if (m + mend > M) or (m + breite > M):
                break
            m = np.int(m) - 1

            r = muw + sigmaw * np.random.randn(hoehe, breite)
            winkelvar = sigmawinkel * np.random.randn(1,1)
            r = np.around(r + winkelvar)
            r[np.where(r < 1)] = 1
            r = r.astype(int)

            for o in range(0, r.shape[0]):
                S[m:m+breite, r[o-1]] = S[m:m+breite, r[o-1]] -1

        m = 1
        while m < M:

            breite = np.around(breite2a + (breite2b - breite2a) * np.random.rand(1,1))
            if breite < 1:
                breite = 1
            breite = np.int(breite)

            mstep = np.around(mu2a + sigma2a * np.random.rand(1,1))
            m = m + mstep
            if (m + mend > M) or (m + breite > M):
                break
            m = np.int(m) - 1

            r = muw + sigmaw * np.random.randn(hoehe, breite)
            winkelvar = sigmawinkel*np.random.randn(1,1)
            r = np.around(r + winkelvar)
            r[np.where(r < 1)] = 1
            r = r.astype(int)

            for o in range(0, r.shape[0]):
                #print(r[0-1])
                S[m:m+breite, r[o-1]] = S[m:m+breite, r[o-1]] -1
        

        iS = iradon(S, theta=theta, circle=False, filter='hamming', interpolation='cubic')
        iSmax = np.max( np.max(np.abs(iS)) ) 
        iS = iS / iSmax
        np.save(pathForSaving+"textureDirectionPlot_{0:05d}".format(cntr), iS)
        groundTruth[cntr] = winkelwerte[k] / 1000 * 60 # Angle in Arc Minute
        cntr +=  1


np.save(pathForLabel+"groundTruth_{0:02d}".format(part),  groundTruth)