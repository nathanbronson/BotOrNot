"""
an app to identify AI-written essays
"""
import asyncio

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
from BotOrNot.main_module import files_to_essays, files_to_essays_gen, text_to_essay, files_to_essays_gen_multi
from functools import partial

def p(*args):
    pass

class BotorNot(toga.App):
    def __init__(self, *args, **kwargs):
        super(BotorNot, self).__init__(*args, **kwargs)
        self.files = []
        self.essays = []
        self.is_analyzing = False
        self.has_files = False
        self.featured_analysis = None
        self.old_files = 0

    def startup(self):
        """
        Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """
        self.main_window = toga.MainWindow(title=self.formal_name)
        
        self.render_main_screen()
        self.things = toga.Group("Things")
        self.select_files_command = toga.Command(
            self.select_files,
            text="Select Files",
            #tooltip="Perform action 0",
            group=self.things
        )
        self.analyze_command = toga.Command(
            self.background_analyze,
            text="Analyze",
            #tooltip="Perform action 1",
            group=self.things
        )
        self.commands.add(self.select_files_command, self.analyze_command)
        self.main_window.toolbar.add(self.select_files_command, self.analyze_command)
        self.main_window.show()
    
    def render_main_screen(self, left=True, right=True):
        print("renderring")
        if left:
            self.left_split = toga.SplitContainer(direction=toga.SplitContainer.HORIZONTAL)
            self.data = [(i["name"], i["predicted_class"]) for i in self.essays]
            self.left_list = toga.Table(headings=["Name", "Classification"], data=self.data, on_select=self.show_analysis)#reset to have buttons to analysis
            self.left_text = toga.MultilineTextInput(placeholder="Enter Text to Analyze")
            self.left_split.content = [(self.left_list, 2), (self.left_text, 1)]
            self.left_container = self.left_split
        if right:
            if self.is_analyzing:
                self.right_content = toga.ProgressBar(max=len(self.files), value=len(self.essays) - self.old_files, style=Pack(flex=1, padding=5, alignment=CENTER))
                self.right_container = toga.Box()
                self.right_container.add(self.right_content)
            else:
                self.right_content = toga.widgets.webview.WebView(style=Pack(flex=1))#toga.Box(style=Pack(direction=COLUMN, padding_top=50))#add webview
                if self.featured_analysis is not None:
                    self.right_content.set_content("https://www.example.com", self.featured_analysis)
                self.right_container = toga.Box()
                self.right_container.add(self.right_content)
        self.split = toga.SplitContainer(direction=toga.SplitContainer.VERTICAL)
        self.split.content = [(self.left_container, 1), (self.right_container, 2)]
        self.main_window.content = self.split
        self.main_window.content.refresh()

    async def select_files(self, *args):
        self.files = await self.main_window.open_file_dialog(
            "Select Files",
            file_types=["txt", "text", "pdf", "doc", "docx", "htm", "html", "pptx", "ppt", "rtf"],
            multiselect=True
        )
        self.files = [str(i) for i in self.files]
        self.analyze_button = toga.Button(
            "Analyze",
            on_press=self.analyze,
            style=Pack(padding=5)
        )
        self.main_window.toolbar.add(self.analyze_command)
        self.has_files = True
    
    async def analyze(self, *args, **kwargs):
        if self.has_files:
            self.is_analyzing = True
            self.render_main_screen()
            #self.essays = []
            iterator = files_to_essays_gen(self.files)
            while True:
                await asyncio.sleep(.1)
                try:
                    ne = next(iterator)
                    if ne["text"] not in [i["text"] for i in self.essays]:
                        self.essays.append(ne)
                except StopIteration:
                    break
                self.render_main_screen()
            self.has_files = False
            self.is_analyzing = False
        if self.left_text.value != "":
            self.is_analyzing = True
            self.old_files += 1
            self.essays.append(text_to_essay(str(self.left_text.value)))
            self.left_text.clear()
            self.is_analyzing = False
        self.old_files += len(self.files)
        self.files = []
        self.render_main_screen()
    

    async def background_analyze(self, widget, **kwargs):
        self.add_background_task(self.analyze)
    
    def show_analysis(self, table, row):
        if not self.is_analyzing:
            analysis = list(filter(lambda e: e["name"] == row.name, self.essays))[0]
            self.featured_analysis = analysis["html_report"]
            self.render_main_screen(left=False)


def main():
    return BotorNot()
