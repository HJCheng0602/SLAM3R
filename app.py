import argparse
from slam3r.demo.app_offline import main_offline
from slam3r.demo.app_online import main_online
def get_args_parser():
    parser = argparse.ArgumentParser(description="A demo for our SLAM3R")
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default="./tmp", help="value for tempfile.tempdir")
    parser.add_argument("--online", action='store_true', help="whether to use the online demo app(mutually exclusive for the offline demo)")
    parser.add_argument("--offline", action="store_true", help="whether to use the offline demo app(mutually exclusice for the online demo)")

    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.online and args.offline:
        print("You cannot use both modes at the same time.")
    elif args.online == True and args.offline == False:
        print("start the online mode")
        main_online(parser)
    elif args.online == False and args.offline == True:
        print("start the offline mode")
        main_offline(parser)
    else:
        print("Please choose '--online' or '--offline' as your arg")
        