#!/usr/bin/env sh

python ./src/xttmp/main.py || {
    echo "---------------------------------------"
    echo "程序运行出错，按回车键退出..."
    read -r
    exit 1
}