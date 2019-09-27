FROM ubuntu:16.04

ARG build_env

WORKDIR /usr/share2host

COPY print_current_dir.sh /usr/share2host

EXPOSE 8080

RUN ["chmod","+x","/usr/share2host/print_current_dir.sh"]

CMD bash /usr/share2host/print_current_dir.sh ; sleep infinity