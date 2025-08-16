import os
import time
import argparse
import open3d as o3d
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue

class RealtimePLYViewer:
    def __init__(self, ply_file_path):
        self.ply_file_path = os.path.abspath(ply_file_path)
        self.is_paused = False
        self.is_running = True
        
        # 使用 Queue 来进行线程间通信
        self.update_queue = Queue()

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="realtime ply viewer")
        
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        self.vis.register_key_callback(ord('P'), self.toggle_pause)
        self.vis.register_key_callback(ord('S'), self.save_screen)
        
        self.observer = Observer()
        self.event_handler = self.PLYFileHandler(self)
        self.observer.schedule(
            self.event_handler,
            os.path.dirname(self.ply_file_path),
            recursive=False
        )

    class PLYFileHandler(FileSystemEventHandler):
        def __init__(self, viewer_instance):
            self.viewer = viewer_instance
            self.last_mtime = 0

        def on_modified(self, event):
            # 在另一个线程中运行
            if os.path.abspath(event.src_path) == self.viewer.ply_file_path:
                current_mtime = os.path.getmtime(self.viewer.ply_file_path)
                if current_mtime > self.last_mtime:
                    self.last_mtime = current_mtime
                    # 放入更新请求到队列中
                    self.viewer.update_queue.put(True)
                    print("Update event triggered. Pushed to queue.")

    def toggle_pause(self, vis):
        self.is_paused = not self.is_paused
        status = "paused" if self.is_paused else "running"
        print(f"[{status}] - PLY view has been {status}")
        return False

    def save_screen(self, vis):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        vis.capture_screen_image(filename)
        print(f"save the screenshot: {filename}")
        return False

    def update_geometry(self):
        if self.is_paused:
            return

        print(f"the ply file has been updated...")
        retry_count = 5
        delay_s = 0.1
        for i in range(retry_count):
            try:
                new_pcd = o3d.io.read_point_cloud(self.ply_file_path)
                if len(new_pcd.points) > 0:
                    self.pcd.points = new_pcd.points
                    self.pcd.colors = new_pcd.colors
                    self.pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                    self.vis.update_geometry(self.pcd)
                    self.vis.reset_view_point(True)
                    self.vis.update_renderer() # 强制渲染
                break
            except PermissionError as pe:
                print(f"PermissionError: File is locked. Retrying... ({i+1}/{retry_count})")
                time.sleep(delay_s)
            except Exception as e:
                print(f"Error reading PLY file: {e}")
                break
        else:
            print(f"Failed to read file after {retry_count} retries.")

    def run(self):
        self.observer.start()
        print(f"The real-time viewer has started, monitoring file: {self.ply_file_path}")
        self.update_geometry() # 初始加载
        
        # 使用非阻塞循环来替代 vis.run()
        while self.is_running:
            # 1. 处理窗口事件
            if not self.vis.poll_events():
                self.is_running = False
                break
            
            # 2. 检查队列中是否有更新请求
            if not self.update_queue.empty():
                self.update_queue.get() # 取出请求
                self.update_geometry() # 在主线程中更新
            
            # 3. 避免 CPU 过高占用
            time.sleep(0.01)

        self.observer.stop()
        self.observer.join()
        self.vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime PLY viewer")
    parser.add_argument("--ply_file", type=str, default="results/realtime/realtime_save.ply", help="PLY file path")
    args = parser.parse_args()
    viewer = RealtimePLYViewer(args.ply_file)
    viewer.run()