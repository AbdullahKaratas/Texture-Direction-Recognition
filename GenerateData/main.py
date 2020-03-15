import numpy as np
from skimage.transform import radon, iradon
import os


def main(N = 2001, sizeImage=301, numberOfAngleSteps=25, runSize = 100):

    cntr =  0   # Numbering for texture direction
    run = 0
    M = np.int( np.around(sizeImage*np.sqrt(2)) ) 

    # Fixed Value
    thetaArea = 1
    if N % 2 == 0:
        N +=  1
    
    if numberOfAngleSteps % 2 == 0:
        numberOfAngleSteps += 1

    if runSize == 0:
        runSize += 1
    
    theta = np.linspace( -thetaArea, thetaArea, N)

    AngleValue = np.linspace(-100, 100, numberOfAngleSteps)
    dataSurface = np.zeros( (AngleValue.size*runSize, sizeImage, sizeImage))
    groundTruth = np.zeros((AngleValue.size*runSize,)) 

    while run < runSize:
        for k in range(0, AngleValue.size):
            S = np.zeros( (M, N) )

            # Angle Area 1 (width of Radontrafo)
            sigmaw = 50
            muw = (N+1) / 2 + AngleValue[k]
            sigmawinkel = 10 # Variation of the Angle Area

            # rho (Distance of  Grooves)
            # Fix Values
            mua = 10
            sigmaa = 100
            mu2a = 5
            sigma2a = 50
            # Width of Grooves
            # Fix Values
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
            iS = (iS - np.min(np.min(iS))) / (np.max(np.max(iS)) - np.min(np.min(iS))) # Norm from 0 to 1
            dataSurface[cntr, :, :] = iS
            groundTruth[cntr] = AngleValue[k] / 1000 * 60 # Angle in Arc Minute
            cntr +=  1
            print(cntr)
        run += 1

    return dataSurface, groundTruth

if __name__ == '__main__':
    dataSurface, groundTruth = main( numberOfAngleSteps=25, runSize = 0 )
    np.save("data",  dataSurface)
    np.save("groundTruth", groundTruth)
