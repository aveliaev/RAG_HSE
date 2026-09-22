FROM alpine:3.20
RUN apk add --no-cache openvpn
# block-outside-dns — опция только для Windows, на Linux openvpn на неё ругается
CMD ["openvpn", "--config", "/vpn/client.ovpn", "--pull-filter", "ignore", "block-outside-dns", "--verb", "1"]
