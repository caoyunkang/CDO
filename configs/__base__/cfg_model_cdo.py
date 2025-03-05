from argparse import Namespace
class cfg_model_cdo(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_t = Namespace()
		self.model_t.name = 'mshrnet32'
		self.model_t.kwargs = dict(pretrained=True, checkpoint_path='',strict=False)

		self.model_s = Namespace()
		self.model_s.name = 'mshrnet32'
		self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)

		self.model = Namespace()
		self.model.name = 'cdo'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=False,
								 model_t=self.model_t, model_s=self.model_s)
