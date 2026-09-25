from gym.envs.registration import register

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv


def __getattr__(name):
    # Selenium belongs to the optional real-browser demo, not text simulation.
    if name == 'WebAgentSiteEnv':
        from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
        return WebAgentSiteEnv
    raise AttributeError(name)

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)
