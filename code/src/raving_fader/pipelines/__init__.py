from raving_fader.pipelines.pipeline_rave import RAVEPipeline
from raving_fader.pipelines.pipeline_ravelight import RAVELightPipeline
from raving_fader.pipelines.pipeline_rave_cond import CRAVEPipeline
from raving_fader.pipelines.pipeline_faderave import FadeRAVEPipeline

pipelines = {
    "rave": RAVEPipeline,
    "ravelight": RAVELightPipeline,
    "crave": CRAVEPipeline,
    "faderave": FadeRAVEPipeline
}
