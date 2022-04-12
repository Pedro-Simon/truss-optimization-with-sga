# coding: utf-8

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.clock import Clock
import os

from pysga import sga
from pysga.configure import ParamsSGA, ObjectiveFunc


MODULE_DIR = os.path.dirname(__file__)


class SGALayout(BoxLayout):
    result = ObjectProperty()
    screen_manager = ObjectProperty()
    get_from_file = ObjectProperty()

    def __init__(self, **kwargs):
        super(SGALayout, self).__init__(**kwargs)
        Clock.schedule_once(self.binds_events)

    def binds_events(self, e):
        self.get_from_file.bind(active=self.on_checkbox_active)

    def optimize(self):
        try:
            x, y = sga.run(sga=self.configure_sga(), fobj=self.configure_function())
            texto = '  Minimum point:\n  %s\n  Optimal value:\n  %s' % (str(x), str(y))
        except:
            texto = '  SGA Error\n  Verify parameters and objective function'
        self.result.text = texto

    def configure_sga(self):
        alfamin = float(self.ids['alfamin'].text)
        alfainitial = float(self.ids['alfainitial'].text)
        iteratios = int(self.ids['iterations'].text)
        global_ratio = float(self.ids['global_ratio'].text)
        population = int(self.ids['population'].text)
        search_group_ratio = float(self.ids['search_group_ratio'].text)
        n_pertubed = int(self.ids['n_pertubed'].text)
        tournament = int(self.ids['tournament'].text)
        small_value = float(self.ids['small_value'].text)

        return ParamsSGA(AlfaMin=alfamin, AlfaInitial=alfainitial, NIterations=iteratios,
                        GlobalIterationsRatio=global_ratio, PopulationSize=population,
                        SearchGroupRatio=search_group_ratio, NPerturbed=n_pertubed,
                        TournamentSize=tournament, SmallValue=small_value)

    def configure_function(self):
        lowers = self.get_list('lowers')
        uppers = self.get_list('uppers')
        if self.get_from_file.active:
            try:
                from fobj_function import fobj
            except ImportError:
                from pysga.fobj_function import fobj
            fobj_func = fobj
        else:
            fobj_func = self.get_fobj_function()
        fobj = ObjectiveFunc(inf=lowers, sup=uppers)
        fobj.evaluate = fobj_func
        return fobj

    def get_list(self, name):
        return eval(self.ids[name].text)

    def get_fobj_function(self):
        func_text = []
        function_full_text = self.ids.fobj.text.split('\n')
        simple_style = False
        ident = False
        ident_txt = ''
        replaces = {'y[]': 'y[:, 0]',
                    'x[': 'x[:, '}

        if 'use simple style' == function_full_text[0]:
            simple_style = True
            del function_full_text[0]
            func_text.append('    y = np.zeros((x.shape[0], 1))\n')
        for line in function_full_text:
            if simple_style:
                if line == '{':
                    ident = True
                    ident_txt += '    '
                    continue
                elif line == '}':
                    ident = False
                    ident_txt = ident_txt[4:]
                    continue
                if ident:
                    line = '%s%s' % (ident_txt, line)
                for expression in replaces.keys():
                    line = line.replace(expression, replaces[expression])
            func_text.append('    %s\n' %line)
        func_text.insert(0, '    import numpy as np\n')
        func_text.insert(0, 'def fobj(x):\n')
        func_text.append('    return y')
        func_text = """{}""".format(''.join(func_text))
        code = compile(func_text, '<string>', 'exec')
        exec_code = {}
        exec(code, exec_code)
        return exec_code['fobj']

    def on_checkbox_active(self, checkbox, value):
        module_fobj_file = os.path.join(MODULE_DIR, 'fobj_function.py')
        if value:
            try:
                with open('fobj_function.py', 'r') as f:
                    fobj_text = f.readlines()
                    self.ids.fobj.text = ''.join(fobj_text)
            except FileNotFoundError:
                with open(module_fobj_file, 'r') as f:
                    fobj_text = f.readlines()
                    self.ids.fobj.text = ''.join(fobj_text)
        else:
            self.ids.fobj.text = self.ids.fobj.text.replace('\n    ', '\n')
            self.ids.fobj.text = self.ids.fobj.text.replace('def fobj(x):', '# def fobj(x):')
            self.ids.fobj.text = self.ids.fobj.text.replace('import numpy as np', '# import numpy as np')
            self.ids.fobj.text = self.ids.fobj.text.replace('return y', '# return y')


class SearchGroupAlgorithmApp(App):
    pass


if __name__ == '__main__':
    from kivy.config import Config
    Config.set('graphics', 'width', '500')
    Config.set('graphics', 'height', '600')
    janela = SearchGroupAlgorithmApp()
    janela.run()
