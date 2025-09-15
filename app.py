import argparse
from slam3r.app.app_offline import main_offline
from slam3r.app.app_online import main_online

def get_args_parser():
    parser = argparse.ArgumentParser(description="A demo for our SLAM3R")
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--server_port", type=int, default=None, 
                        help=("will start gradio app on this port (if available). If None, will search for an available port starting at 7860."))
    parser.add_argument("--viser_server_port", type=int, default=8080, 
                        help="will start viser server on this port (if available), default is 8080")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default="./tmp", help="value for tempfile.tempdir")
    parser.add_argument("--online", action='store_true', help="whether to use the online demo app")
    
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    
    if args.online:
        print("start the online mode")
        main_online(parser)
    else:
        print("start the offline mode")
        main_offline(parser)
        