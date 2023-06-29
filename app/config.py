class Config(object):
    DEBUG = True
    TESTING = False


class DevelopmentConfig(Config):
    SECRET_KEY = 'Francesco'
    OPENAI_API_KEY = 'api-key'


config = {
    'development': DevelopmentConfig,
    'testing': DevelopmentConfig,
    'production': DevelopmentConfig
}
