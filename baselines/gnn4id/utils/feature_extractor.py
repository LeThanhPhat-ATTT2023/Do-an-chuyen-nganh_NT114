"""nfstream-based PCAP → CSV extractor (GNN4ID style, max 20 pkts/flow)."""
from __future__ import annotations
import pandas as pd
from nfstream import NFPlugin, NFStreamer


class _PacketCapture(NFPlugin):
    """Captures per-packet payload hex + metadata for up to `limit` packets."""

    def on_init(self, packet, flow):
        raw = packet.ip_packet
        flow.udps.pkt_hex = [raw[20:].hex() if raw else ""]
        flow.udps.pkt_delta = [0]
        flow.udps.pkt_dir = [packet.direction]
        flow.udps.pkt_ip_size = [packet.ip_size]
        flow.udps.pkt_transport_size = [packet.transport_size]
        flow.udps.pkt_payload_size = [packet.payload_size]
        flow.udps.syn = [packet.syn]
        flow.udps.cwr = [packet.cwr]
        flow.udps.ece = [packet.ece]
        flow.udps.urg = [packet.urg]
        flow.udps.ack = [packet.ack]
        flow.udps.psh = [packet.psh]
        flow.udps.rst = [packet.rst]
        flow.udps.fin = [packet.fin]

    def on_update(self, packet, flow):
        if flow.bidirectional_packets >= self.limit:
            flow.expiration_id = -1
            return
        raw = packet.ip_packet
        flow.udps.pkt_hex.append(raw[20:].hex() if raw else "")
        flow.udps.pkt_delta.append(packet.delta_time)
        flow.udps.pkt_dir.append(packet.direction)
        flow.udps.pkt_ip_size.append(packet.ip_size)
        flow.udps.pkt_transport_size.append(packet.transport_size)
        flow.udps.pkt_payload_size.append(packet.payload_size)
        flow.udps.syn.append(packet.syn)
        flow.udps.cwr.append(packet.cwr)
        flow.udps.ece.append(packet.ece)
        flow.udps.urg.append(packet.urg)
        flow.udps.ack.append(packet.ack)
        flow.udps.psh.append(packet.psh)
        flow.udps.rst.append(packet.rst)
        flow.udps.fin.append(packet.fin)


def extract_pcap_to_csv(
    pcap_path: str,
    out_csv: str,
    label: str,
    max_pkts: int = 20,
) -> None:
    """Run nfstream on one PCAP and write a labelled CSV.

    Args:
        pcap_path: Path to input .pcap file.
        out_csv:   Destination CSV path.
        label:     Class name to store in the ``label`` column.
        max_pkts:  Maximum bidirectional packets captured per flow (default 20).
    """
    streamer = NFStreamer(
        source=pcap_path,
        udps=_PacketCapture(limit=max_pkts),
        statistical_analysis=True,
    )
    streamer.to_csv(path=out_csv, flows_per_file=0, columns_to_anonymize=[])
    df = pd.read_csv(out_csv)
    df["label"] = label
    df.to_csv(out_csv, index=False)
