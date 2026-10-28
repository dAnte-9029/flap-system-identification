import unittest
import numpy as np
from scipy.integrate import solve_ivp
from phase_forecast import propagate


class DiagnosticChecks(unittest.TestCase):
    def test_phase_integral_against_independent_solver(self):
        c=np.array([3.6,23.4,-5.1]);f,p=propagate(3.2,.4,.67,2.,c)
        ref=solve_ivp(lambda t,x:[c[0]+c[1]*.67+c[2]*x[0],2*np.pi*x[0]],
                      [0,2],[3.2,.4],rtol=1e-11,atol=1e-12).y[:,-1]
        np.testing.assert_allclose([f,p],ref,atol=1e-9,rtol=1e-9)

    def test_zoh_segment_composition(self):
        c=[3.6,23.4,-5.1]
        f,p=propagate(3.2,.4,.67,.3,c)
        np.testing.assert_allclose(propagate(f,p,.67,.7,c),propagate(3.2,.4,.67,1.,c),atol=1e-12)

    def test_zero_frequency_drift_coefficient(self):
        f,p=propagate(3.,.2,.5,.7,[1.,2.,0.])
        self.assertAlmostEqual(f,4.4)
        self.assertAlmostEqual(p,.2+2*np.pi*(3*.7+.7**2))

    def test_phase_gauge_equivalence(self):
        # Unknown mechanical zero rotates coefficients without changing predictions.
        phase=np.linspace(0,2*np.pi,1024,endpoint=False);delta=.71
        for h in [1,2,3]:
            a,b=1.2,-.7
            ap=a*np.cos(h*delta)+b*np.sin(h*delta)
            bp=b*np.cos(h*delta)-a*np.sin(h*delta)
            np.testing.assert_allclose(a*np.sin(h*phase)+b*np.cos(h*phase),
                                       ap*np.sin(h*(phase+delta))+bp*np.cos(h*(phase+delta)),atol=3e-15)

    def test_closed_loop_gain_nonuniqueness_example(self):
        # Illustrative exact counterexample, NOT a rank test of the actual logs.
        z=np.linspace(-1,1,101);phi=np.linspace(0,2*np.pi,101);u=.3*z
        g=1+.5*np.sin(phi);dg=.7*np.cos(phi);c=z**2
        np.testing.assert_allclose(c+g*u,(c-dg*.3*z)+(g+dg)*u,atol=1e-15)
        self.assertGreater(np.max(abs(dg)),.69)


if __name__=='__main__':unittest.main()
