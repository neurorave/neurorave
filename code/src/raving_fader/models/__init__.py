from raving_fader.models.rave.rave import RAVE
from raving_fader.models.rave.cond_rave import CRAVE
from raving_fader.models.rave.ravelight import RAVELight
from raving_fader.models.fader.faderave import FadeRAVE

models = {"rave": RAVE, "ravelight": RAVELight, "crave": CRAVE, "faderave": FadeRAVE}
