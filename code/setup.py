from igraph import Graph
import geopandas as gpd
import numpy as np
import pandas as pd

from mobilkit.umni import Project
import mobilkit.utils as U
from mobilkit.geo import CRS_DEG

P = Project('..')

class City:
    def __init__(self, geocode, name=None, root=P.data):
        self.geocode = geocode
        self.name = name if isinstance(name, str) else geocode.split(',')[0]
        self.label = self.name.lower().replace(' ', '_')
        self.root = U.mkdir(f'{root}/{self.label}')

    def __repr__(self):
        return f'City({self.name})'

    def load(self, layer, crs=CRS_DEG):
        if hasattr(self, layer):
            return getattr(self, layer)
        df = pd.read_parquet(self.root / f'{layer}.parquet')
        if 'geometry' in df.columns:
            geom = gpd.GeoSeries.from_wkb(df.geometry)
            df = gpd.GeoDataFrame(df, geometry=geom, crs=crs)
        setattr(self, layer, df)
        return df

    def save(self, fname, df):
        setattr(self, fname, df)
        df.to_parquet(self.root / f'{fname}.parquet')


class Pednet:
    def __init__(self, city, query=None, name='', od_sample_id='', E=None):
        # load prepared data for this city
        V = city.load('full_pednet_nodes')
        if E is None:
            E = city.load('simple_sidewalk_pednet_edges')
        odpt2V = city.load('odpt2node').set_index('odpt_id')
        od = pd.read_parquet(city.root / f'od_samples/{od_sample_id}.parquet')
        self.city = city
        self.name = name
        E = E.query(query) if isinstance(query, str) else E
        V = V.loc[list(set(E['src_vid']) | set(E['trg_vid']))].sort_index()
        G = Graph()
        G.add_vertices(V.shape[0], {'vid': V.index})
        vid2idx = pd.Series(range(V.shape[0]), index=V.index)
        end_pts = list(
            zip(vid2idx.loc[E['src_vid']], vid2idx.loc[E['trg_vid']]))
        E2 = E.rename_axis('id').drop(columns=['id', 'geometry'],
                                      errors='ignore').reset_index()
        G.add_edges(end_pts, {x: E2[x].values for x in E2.columns})
        V['cid'] = G.clusters().membership
        E = E.merge(V['cid'].rename('src_cid'), left_on='src_vid',
                    right_index=True, how='left')
        E = E.merge(V['cid'].rename('trg_cid'), left_on='trg_vid',
                    right_index=True, how='left')
        E['cid'] = E['src_cid'] | E['trg_cid']
        E = E.drop(columns=['src_cid', 'trg_cid']).rename_axis('id')

        odpt2V['dist'] = odpt2V['dist_odpt2cp'] + odpt2V['dist_cp2node']
        od = (od.drop_duplicates().reset_index()
              .merge(od.groupby(['orig_odpt', 'dest_odpt']).size()
                     .rename('n_ods'), on=('orig_odpt', 'dest_odpt'))
              .merge(odpt2V[['vid', 'dist']].rename(columns=lambda x: x + '_o'),
                     left_on='orig_odpt', right_index=True)
              .merge(odpt2V[['vid', 'dist']].rename(columns=lambda x: x + '_d'),
                     left_on='dest_odpt', right_index=True)
              .assign(dist=lambda df: df.pop('dist_o') + df.pop('dist_d'))
              .astype({'vid_o': np.int32, 'vid_d': np.int32})
              .set_index('od_id').sort_index())
        self.V, self.E, self.G, self.vid2idx, self.od = V, E, G, vid2idx, od

    def __repr__(self):
        return f'Pednet("{self.name}" in {self.city})'

    def get_sp(self, o, d):
        oid, did = self.vid2idx.loc[o], self.vid2idx.loc[d]
        sp = self.G.get_shortest_paths(oid, did, 'len', output='epath')[0]
        edges = [e['id'] for i, e in enumerate(self.G.es) if i in sp]
        return {'vid_o': o, 'vid_d': d, 'edges': edges,
                'd_V2V': self.E.loc[edges]['len'].sum()}
